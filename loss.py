import tensorflow as tf
from tensorflow.keras import backend as K


def projection_residual(z, U):
	loss = tf.norm(z - tf.dot(tf.dot(U, U.T), z))
	loss = tf.cast(tf.matmul(loss, loss), dtype=tf.float64)

	return loss

def reconst_loss(x_true, x_reconst):
	return tf.reduce_mean(tf.square(x_true - x_reconst))

def ae_loss(x_true, x_reconst, z_conv, z_se, theta, lambda1=0.5, lambda2=15, lambda3=1):
	x_reconst = tf.cast(x_reconst, dtype=tf.float64)
	reconst_loss = tf.reduce_sum(tf.square(x_true - x_reconst))
	self_expr_loss = tf.cast(tf.reduce_sum(tf.square(z_se - z_conv)), dtype=tf.float64)
	norm = tf.norm(theta, keepdims=True)

	penalty = tf.cast(tf.matmul(norm, norm), dtype=tf.float64)
	# print(reconst_loss.dtype, self_expr_loss.dtype, penalty.dtype)
	loss = lambda1 * reconst_loss + lambda2 * self_expr_loss + lambda3 * penalty

	return [loss, reconst_loss, self_expr_loss, penalty]

def loss_D(z, _z, _U, kcluster, _m, epsilon=0):
	loss = 0
	for k in range(kcluster):
		m = z[k].shape[1]			# z[k] 列向量构成的矩阵，m为向量数即样本数
		loss = loss + projection_residual(z[k], _U[k])/m + max(epsilon - projection_residual(_z[k], _U[k]), 0)/m

	loss = loss/kcluster

	return loss
