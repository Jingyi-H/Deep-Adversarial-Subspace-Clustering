import numpy as np
import tensorflow as tf
import scipy.io as sio
import loss
from model import ConvAE



def train_G(train_data, batch_size=72, input_shape=[32,32,1], epoch_num=10):
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
    optimizer = tf.optimizers.Adam(lr=0.1)

    train_size = len(list(train_data))
    train_data = train_data.batch(batch_size)

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
                str(epoch+1), str(float(rec_loss)*0.01), str(reconst_loss.numpy()), str(self_expr_loss.numpy()), str(penalty.numpy().ravel()[0])))

            # if (batch+1)%10 == 0:
            #     print("epoch[{}]: batch-{} loss={}".format(str(epoch+1), str(batch+1), str(float(rec_loss))))
    print(conv_ae.layers[1].get_weights()[0])

coil20 = sio.loadmat('data/COIL20.mat')
img = coil20['fea']
label = coil20['gnd']
img = np.reshape(img,(img.shape[0],32,32,1))

# train_db = tf.convert_to_tensor(img, dtype=tf.float32)
train_db = tf.data.Dataset.from_tensor_slices((img, img))
# print(type(train_db))
# train_db = train_db.batch(72)

# print(train_db)
# print(list(train_db.as_numpy_iterator()))
train_G(train_db, epoch_num=10, batch_size=1440)
