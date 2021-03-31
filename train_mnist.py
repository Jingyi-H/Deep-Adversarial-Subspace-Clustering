import numpy as np
import tensorflow as tf
import os
import h5py
import scipy.io as sio
from tensorflow.keras import datasets

import loss
from model import Mnist_ConvAE
from model import DASC
import utils
from sklearn.cluster import SpectralClustering

# block warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train(train_data, batch_size=1000, input_shape=[28,28,1], epoch_num=10, pre_train_epoch=10,
          k=10, d_iter_num=5, r=10, alpha=0.95, g_lr=1e-3, d_lr=2e-4, save_dir='model'):

    inputs = tuple([batch_size] + input_shape)

    dasc = DASC(Mnist_ConvAE, input_shape=inputs, batch_size=batch_size, kcluster=k)
    dasc.initialize(train_data, pre_train_epoch=pre_train_epoch, lambda2=0.1, lambda3=1.0)

    print("Start Training...")
    g_var = dasc.conv_ae.trainable_variables
    # d_var = dasc._U
    d_optim = tf.optimizers.Adam(lr=d_lr)
    g_optim = tf.optimizers.Adam(lr=g_lr)
    for epoch in range(epoch_num):
        for (batch, (x_batch, y_batch)) in enumerate(train_data):
            with tf.GradientTape(watch_accessed_variables=False) as gtape:
                real_z, fake_z = dasc.G(x_batch, alpha=alpha)
                dasc.D(real_z, fake_z, r)
                d_var = [dasc._U[i].trainable_variables[0] for i in range(k)]
                # print(d_var)
                for i in range(d_iter_num):
                    with tf.GradientTape(watch_accessed_variables=False) as tape:
                        tape.watch(d_var)
                        real_proj = dasc.forward(real_z)
                        fake_proj = dasc.forward(fake_z)
                        l_d = loss.L_D(real_z, fake_z, real_proj, fake_proj, k)
                        l_d = l_d + loss.r1(d_var) + loss.r2(d_var)
                        print("\tLoss_D: loss={}\tR1={}\tR2={}".format(str(float(l_d)), str(float(loss.r1(d_var))), str(float(loss.r2(d_var)))))
                    d_grads = tape.gradient(l_d, d_var)
                    d_optim.apply_gradients(zip(d_grads, d_var))
                # print("Epoch[{}]: D loss={}".format(str(epoch+1), str(l_d)))

                # D_loss = loss.L_D(cluster, d_var, k, dasc._m)
                proj_fake = dasc.forward(fake_z)
                # with tf.GradientTape() as tape:
                gtape.watch(g_var)
                x_reconst = dasc.conv_ae(x_batch)
                z_conv = dasc.conv_ae.z_conv
                z_se = dasc.conv_ae.z_se

                theta = dasc.conv_ae.layers[1].get_weights()[0]
                theta = np.reshape(theta, [batch_size, batch_size])
                rec_loss, reconst_loss, self_expr_loss, penalty = loss.ae_loss(x_batch, x_reconst, z_conv, z_se, theta, lambda2=0.1, lambda3=1)
                G_loss = rec_loss + loss.L_r(fake_z, proj_fake, k)
                # calculate accuracy of prediction
                # theta = utils.theta_normalize(theta)
                affinity = 0.5 * (np.abs(theta) + np.abs(theta.T))
                sc = SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity='precomputed', n_init=100, assign_labels='kmeans')
                y_hat = sc.fit_predict(affinity)
                y_hat = y_hat.astype(int)
                y_true = tf.reshape(y_batch, y_hat.shape)
                acc = utils.cluster_accuracy(y_true, y_hat)

                g_grads = gtape.gradient(G_loss, g_var)

                g_optim.apply_gradients(zip(g_grads, g_var))

            print("Epoch[{}/{}]: G_loss={}\tD_loss={}\tpenalty={}\tacc={}".format(str(epoch+1), str(epoch_num), str(float(G_loss)), str(float(l_d)), str(float(penalty)), str(float(acc))))
            # print("epoch[{}]: G_loss={}\tD_loss={}\treconst_loss={}\tself_expr_loss={}\tpenalty={}".format(
            #     str(epoch+1), str(float(rec_loss)), str(float(l_d)), str(reconst_loss.numpy()), str(self_expr_loss.numpy()), str(penalty.numpy().ravel()[0])))

    theta = dasc.conv_ae.layers[1].get_weights()[0]
    theta = np.reshape(theta, [batch_size, batch_size])
    print(theta)

    # save_weights of G
    if not os.path.exists('logs'):
        os.makedirs('logs')
    dasc.conv_ae.save_weights('logs/conv_ae_weights.h5')

    return theta

def load_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    for i in range(10):
        idx = np.where(y_train == i)[0][0:100]
        if i == 0:
            x = x_train[idx, :, :]
            y = y_train[idx]
        else:
            x = np.concatenate([x, x_train[idx, :, :]], axis=0)
            y = np.concatenate([y, y_train[idx]], axis=0)

    x = np.reshape(x, [-1, 28, 28, 1])
    return x, y

x_train, y_train = load_mnist()
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_db = train_db.batch(1000).shuffle(10)
train(train_db, epoch_num=20, batch_size=1000, pre_train_epoch=1000, alpha=0.9, g_lr=1e-3, d_lr=2e-4)


