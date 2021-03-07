import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, Input
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import Layer
from sklearn.cluster import SpectralClustering
import numpy as np
from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

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
		z = self.self_expressive(z)
		self.z_se = z									# self_expressive_z = theta*z
		# print("z_se:", z.shape)
		z = tf.reshape(z, [self.batch_size, 1, 3840])
		x = self.decoder(z)

		return x


class DASC(object):
	def __init__(self, input_img, input_shape=[32,32,1], batch_size=None, kclusters=20):
		super(DASC, self).__init__()
		self.img = input_img
		self.input_shape = input_shape
		self.batch_size = batch_size
		self.kclusters = kclusters

	def forward(self, x):
		pass


	def G(self, ae):
		# ae = ConvAE(batch_size=self.batch_size)
		# ae.compile(optimizer='adam', loss=losses.MeanSquaredError())
		# ae.fit(self.img, self.img, epochs=10, shuffle=True, batch_size=self.batch_size, validation_split=0.5)
		theta = tf.reshape(ae.theta, [self.batch_size, self.batch_size]).numpy()
		affinity = 0.5*(theta + theta.T)

		# 构造相似度矩阵，自表达层的参数即kernel的权重
		affinity = np.abs(affinity)
		sc = SpectralClustering(self.kclusters, affinity='precomputed', n_init=100, assign_labels='discretize')
		sc.fit_predict(affinity)



	def D(self):
		pass
