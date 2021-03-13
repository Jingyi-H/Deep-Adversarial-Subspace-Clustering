import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, Input
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer
from sklearn.cluster import SpectralClustering
import numpy as np
import math
from scipy.linalg import qr
# from scipy.sparse.linalg import svds
# from sklearn.preprocessing import normalize

from DASC import utils
from DASC import loss


class Self_Expressive(Layer):
    def __init__(self, batch_size, **kwargs):
        super(Self_Expressive, self).__init__(**kwargs)
        self.batch_size = batch_size

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        self.theta = self.add_weight(name='Theta',
                                      shape=(1, self.batch_size, self.batch_size),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Expressive, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        return K.dot(self.theta, x)

class ConvAE(Model):
    def __init__(self, input_shape=[32,32,1], batch_size=None, learning_rate=1e-3, num_class=20):
        super(ConvAE, self).__init__()

        # self.input_shape = input_shape # [32, 32, 1] i.e. 32x32x1 单通道图像
        self.batch_size = batch_size
        # self.input_img = layers.Input(shape=input_shape)
        self.learning_rate = learning_rate
        # COIL-20的编码器只有一层卷积
        self.encoder = Sequential([
            layers.Conv2D(15, kernel_size=3,
                          activation='relu',
                          input_shape=input_shape,
                          strides=2,
                          padding='SAME',
                          kernel_initializer = tf.keras.initializers.GlorotNormal()),
            layers.Reshape((-1, 3840))
        ])

        self.self_expressive = Self_Expressive(self.batch_size, input_shape=(batch_size, batch_size))
        self.z_conv = None
        self.z_se = None
        self.decoder = Sequential([
            layers.Reshape((16, 16, 15)),
            layers.Conv2DTranspose(15, kernel_size=3,
                                   activation='relu',
                                   # input_shape=input_shape,
                                   strides=2,
                                   padding='SAME',
                                   kernel_initializer = tf.keras.initializers.GlorotNormal())
        ])

    # def build(self):
    # 	inputs = [self.batch_size] + self.input_shape
    # 	inputs = tuple(inputs)
    # 	self.encoder.build(inputs)
    # 	self.self_expressive.build((-1, 3840))
    # 	self.decoder.build((-1, 3840))

    def call(self, x):
        z = self.encoder(x)
        z = tf.reshape(z, [self.batch_size, 3840])	# 整个batch一起训练
        self.z_conv = z
        # print(z)
        z = self.self_expressive(z)
        self.z_se = z									# self_expressive_z = theta*z
        # print("z_se:", z.shape)
        z = tf.reshape(z, [self.batch_size, 1, 3840])
        x = self.decoder(z)

        return x


class DASC(object):
    def __init__(self, input_shape, batch_size=1440, kcluster=20):
        super(DASC, self).__init__()
        self.kcluster = kcluster
        self.conv_ae = ConvAE(batch_size=batch_size, input_shape=input_shape)
        self.conv_ae.build(input_shape=input_shape)

    def forward(self, x):
        pass

    def initialize(self, train_data, pre_train_epoch=10):
        '''
        pre-train generator G without considering D
        :param x:
        :return:
        '''

        variables = self.conv_ae.trainable_variables
        optimizer = tf.optimizers.Adam(lr=2e-4)

        for epoch in range(pre_train_epoch):
            for (batch, (x_batch, x_r_batch)) in enumerate(train_data):
                with tf.GradientTape() as tape:
                    x_reconst = self.conv_ae(x_batch)
                    z_conv = self.conv_ae.z_conv
                    z_se = self.conv_ae.z_se
                    theta = self.conv_ae.layers[1].get_weights()[0]
                    # print(x_reconst.dtype, x_r_batch.dtype)
                    rec_loss, reconst_loss, self_expr_loss, penalty = loss.ae_loss(x_r_batch, x_reconst, z_conv, z_se, theta)
                grads = tape.gradient(rec_loss, variables)
                optimizer.apply_gradients(zip(grads, variables))

            print("epoch[{}]: loss={}\treconst_loss={}\tself_expr_loss={}\tpenalty={}".format(
                str(epoch+1), str(float(rec_loss)), str(reconst_loss.numpy()), str(self_expr_loss.numpy()), str(penalty.numpy().ravel()[0])))


    def G(self, x, theta, z_conv, real_label, alpha=0.8):
        '''
        Cluster Generator
        :param theta: 	Self-Expressive Coefficient Matrix
        :param z_conv:	Representation
        :return:		Generated vectors
        '''
        x_reconst = self.conv_ae(x)
        z_conv = self.conv_ae.z_conv
        z_se = self.conv_ae.z_se
        theta = self.conv_ae.layers[1].get_weights()[0]

        affinity = 0.5*(theta + theta.T)

        # 构造相似度矩阵，自表达层的参数即kernel的权重
        affinity = np.abs(affinity)
        # Spectral Clustering(N-Cut)
        sc = SpectralClustering(self.kcluster, affinity='precomputed', n_init=100, assign_labels='discretize')
        label_pred = sc.fit_predict(affinity)

        cidx = []
        label = []
        gen_data = np.array([])
        # generate fake data
        for k in range(self.kcluster):
            idx = np.where(label_pred == k)[0].tolist()
            cidx.append(idx)
            basis = z_conv[idx][:]
            # clusters.append(z_conv[idx][:])
            m_k = len(idx)						# num of samples in cluster Ck
            m_gen = math.floor(m_k * alpha)		# num of generated samples
            # 生成数据
            z_k = utils.generate_data(z_conv, m_k, m_gen)
            # 生成标签
            l = np.tile(np.array([k]), m_gen).reshape(-1, 1)
            if k == 0:
                label = l
                gen_data = z_k
            else:
                label = np.vstack(l)
                gen_data = np.vstack(z_k)

        return label, gen_data

    def D(self):
        pass
