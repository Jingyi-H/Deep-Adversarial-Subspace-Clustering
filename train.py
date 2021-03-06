import numpy as np
import tensorflow as tf
import scipy.io as sio
import loss
from model import ConvAE



def train_G(train_data, batch_size=72, input_shape=[32,32,1], epoch_num=10):
    inputs = tuple([batch_size] + input_shape)
    conv_ae = ConvAE(batch_size=batch_size)
    conv_ae.build(input_shape=inputs)

    variables = conv_ae.trainable_variables
    optimizer = tf.optimizers.Adam(lr=0.001)

    for epoch in epoch_num:
        for step, (x_batch, x_r_batch) in enumerate(train_data):
            with tf.GradientTape() as tape:
                x_reconst = conv_ae(x_batch)
                print(x_reconst, type(x_reconst))
                z_conv = conv_ae.z_conv
                z_se = conv_ae.z_se
                theta = conv_ae.layers[1].get_weights()[0]

                rec_loss = loss.ae_loss(x_r_batch, x_reconst, z_conv, z_se, theta)

            grads = tape.gradient(rec_loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step % 100 ==0:
                print(epoch, step, float(rec_loss))

coil20 = sio.loadmat('data/COIL20.mat')
img = coil20['fea']
label = coil20['gnd']
img = np.reshape(img,(img.shape[0],32,32,1))

# train_db = tf.convert_to_tensor(img, dtype=tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((img, img))
train_G(train_db, epoch_num=1)
