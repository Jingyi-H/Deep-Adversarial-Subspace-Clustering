import math

import numpy as np
import tensorflow as tf
import os
import h5py
import scipy.io as sio
from sklearn.cluster import SpectralClustering
import loss
from model import ConvAE
from model import DASC
import utils

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train(train_data, batch_size=72, input_shape=[32,32,1], epoch_num=10, k=20, d_iter_num=5, r=30, save_dir='model'):

    inputs = tuple([batch_size] + input_shape)
    print(inputs)
    dasc = DASC(input_shape=inputs, kcluster=20)
    dasc.initialize(train_data)

    g_var = dasc.conv_ae.trainable_variables
    # d_var = dasc._U
    optimizer = tf.optimizers.Adam(lr=2e-4)

    for epoch in range(epoch_num):
        for (batch, (x_batch, x_r_batch)) in enumerate(train_data):
            with tf.GradientTape(watch_accessed_variables=False) as gtape:
                real_z, fake_z = dasc.G(x_batch)
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
                        # print("L_d: {}".format(str(l_d)))
                    grads = tape.gradient(l_d, d_var)
                    # print(tape.watched_variables())
                    optimizer.apply_gradients(zip(grads, d_var))
                # print("Epoch[{}]: D loss={}".format(str(epoch+1), str(l_d)))

                # D_loss = loss.L_D(cluster, d_var, k, dasc._m)
                proj_fake = dasc.forward(fake_z)
                # with tf.GradientTape() as tape:
                gtape.watch(g_var)
                x_reconst = dasc.conv_ae(x_batch)
                z_conv = dasc.conv_ae.z_conv
                z_se = dasc.conv_ae.z_se
                theta = dasc.conv_ae.layers[1].get_weights()[0]
                rec_loss, reconst_loss, self_expr_loss, penalty = loss.ae_loss(x_r_batch, x_reconst, z_conv, z_se, theta)
                G_loss = rec_loss + loss.L_r(fake_z, proj_fake, k)
                grads = gtape.gradient(G_loss, g_var)
                optimizer.apply_gradients(zip(grads, g_var))

            print("epoch[{}]: G_loss={}\tD_loss={}\treconst_loss={}\tself_expr_loss={}\tpenalty={}".format(
                str(epoch+1), str(float(rec_loss)), str(float(l_d)), str(reconst_loss.numpy()), str(self_expr_loss.numpy()), str(penalty.numpy().ravel()[0])))

    theta = dasc.conv_ae.layers[1].get_weights()[0]
    theta = np.reshape(theta, [batch_size, batch_size])
    print(theta)

    # save_weights of G
    if not os.path.exists('logs'):
        os.makedirs('logs')
    dasc.conv_ae.save_weights('logs/conv_ae_weights.h5')

    return theta


def train_ConvAE(train_data, batch_size=1440, input_shape=[32,32,1], epoch_num=10, save_dir='autoencoder'):
    '''

    :param train_data: tf.data.Dataset
    :param batch_size: batch size
    :param input_shape: input shape without batch
    :param epoch_num:
    :return:
    '''

    inputs = tuple([batch_size] + input_shape)
    conv_ae = ConvAE(batch_size=batch_size)
    conv_ae.build(input_shape=inputs)

    variables = conv_ae.trainable_variables
    # print(type(variables), variables)
    optimizer = tf.optimizers.Adam(lr=2e-4)

    for epoch in range(epoch_num):
        for (batch, (x_batch, x_r_batch)) in enumerate(train_data):
            with tf.GradientTape() as tape:
                x_reconst = conv_ae(x_batch)
                z_conv = conv_ae.z_conv
                z_se = conv_ae.z_se
                theta = conv_ae.layers[1].get_weights()[0]
                # print(x_reconst.dtype, x_r_batch.dtype)
                rec_loss, reconst_loss, self_expr_loss, penalty = loss.ae_loss(x_r_batch, x_reconst, z_conv, z_se, theta)
            grads = tape.gradient(rec_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))

            print("epoch[{}]: loss={}\treconst_loss={}\tself_expr_loss={}\tpenalty={}".format(
                str(epoch+1), str(float(rec_loss)), str(reconst_loss.numpy()), str(self_expr_loss.numpy()), str(penalty.numpy().ravel()[0])))

            # if (batch+1)%10 == 0:
            #     print("epoch[{}]: batch-{} loss={}".format(str(epoch+1), str(batch+1), str(float(rec_loss))))
    # print(conv_ae.layers[1].get_weights()[0])
    _z_conv = conv_ae.z_conv.numpy()
    _theta = conv_ae.layers[1].get_weights()[0]
    _theta = np.reshape(_theta, [batch_size, batch_size])

    # conv_ae.save_weights('logs/conv_ae_weights')
    return _z_conv, _theta

def load_data(path='data/COIL20.mat'):
    try:
        print("loading data from {}\{}...".format(os.getcwd(), path))
        coil20 = sio.loadmat(path)
        img = coil20['fea']
        label = coil20['gnd']
        img = np.reshape(img,(img.shape[0],32,32,1))
        return img, label
    except Exception as e:
        print(e)
        return None



def predict():
    pass

img, label = load_data()
img, label = utils.shuffle(img, label)
train_db = tf.data.Dataset.from_tensor_slices((img, img))
train_db = train_db.batch(1440)

label = label.astype(int)
label = label.reshape([1, -1])
print(label)
theta = train(train_db, epoch_num=30, batch_size=1440)
# np.savetxt('theta.txt', theta)
# np.savetxt('label.txt', label)
# theta_T = np.transpose(theta)

# z, theta = train_ConvAE(train_db, epoch_num=10, batch_size=1440)
affinity = np.abs(theta + theta.T)
affinity = np.abs(affinity)
sc = SpectralClustering(20, affinity='precomputed', n_init=100, assign_labels='discretize')
pred = sc.fit_predict(affinity)
pred = pred.astype(int)
pred = utils.best_map(label, pred)
pred = pred.astype(int)

label = label.reshape(pred.shape)
print(np.argwhere(pred-label==0).shape)
# np.savetxt('pred.txt', label_map)
# z, theta = train_ConvAE(train_db, epoch_num=10, batch_size=1440)
