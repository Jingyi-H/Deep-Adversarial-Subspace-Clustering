import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def projection_residual(z, proj):
	# z = tf.cast(z, dtype='float64')
	# proj = tf.cast(proj, dtype='float64')
	loss = tf.norm(z - proj, axis=0)
	loss = loss * loss

	return loss

def reconst_loss(x_true, x_reconst):
	return tf.reduce_mean(tf.square(x_true - x_reconst))

def ae_loss(x_true, x_reconst, z_conv, z_se, theta, lambda1=0.1, lambda2=15, lambda3=1):
	# x_reconst = tf.cast(x_reconst, dtype=tf.float64)
	x_true = tf.cast(x_true, dtype=x_reconst.dtype)
	reconst_loss = tf.reduce_sum(tf.square(x_true - x_reconst))
	self_expr_loss = tf.reduce_sum(tf.square(z_se - z_conv))
	# self_expr_loss = tf.cast(tf.reduce_sum(tf.square(z_se - z_conv)), dtype=tf.float64)
	norm = tf.norm(theta, keepdims=True)

	penalty = tf.matmul(norm, norm)
	# penalty = tf.cast(tf.matmul(norm, norm), dtype=tf.float64)
	# print(reconst_loss.dtype, self_expr_loss.dtype, penalty.dtype)
	loss = lambda1 * reconst_loss + lambda2 * self_expr_loss + lambda3 * penalty

	return [loss, reconst_loss, self_expr_loss, penalty]

def L_r(z, proj, kcluster):
	loss = 0
	for k in range(kcluster):
		# m = z[k].shape[1]										# z[k] 列向量构成的矩阵，m为向量数即样本数
		loss = loss + tf.reduce_mean(projection_residual(z[k], proj[k]))

	loss = loss/kcluster

	return loss

def L_D(real_z, fake_z, real_proj, fake_proj, kcluster, epsilon=0.1):
	# loss = L_r(real_z, real_proj, kcluster) + max(0, epsilon/kcluster - L_r(fake_z, fake_proj, kcluster))
	loss = 0
	for k in range(kcluster):
		m_ = fake_z[k].shape[1]
		L_real = projection_residual(real_z[k], real_proj[k])
		idx = tf.argsort(L_real, axis=-1, direction='ASCENDING')
		L_real = tf.sort(L_real, axis=-1, direction='ASCENDING')
		# z = tf.gather(real_z[k], axis=1, indices=index)
		# proj = tf.gather(real_proj[k], axis=1, indices=index)
		term2 = epsilon * tf.ones([1, m_], dtype=L_real.dtype) - projection_residual(fake_z[k], fake_proj[k])
		loss = loss + tf.reduce_mean(L_real[0:m_]) + tf.reduce_mean(tf.maximum(term2, tf.zeros([1, m_], dtype=L_real.dtype)))

	loss = loss/kcluster
	
	return loss

def r1(u_list, beta1=0.01):
	m = len(u_list)
	R1 = 0
	for i in range(m):
		for j in range(m):
			if i == j:
				continue
			else:
				r = tf.matmul(u_list[i], u_list[j], transpose_a=True)
				r = tf.norm(r, keepdims=True)
				r = r * r
				R1 = R1 + r

	R1 = beta1 * R1
	# print("R1 = ", R1)

	return R1

def r2(u_list, beta2=0.01):
	m = len(u_list)
	R2 = 0
	for i in range(m):
		r = tf.matmul(u_list[i], u_list[i], transpose_a=True)
		I = tf.Variable(np.eye(r.shape[0]), dtype=r.dtype)
		r = tf.norm(r - I, keepdims=True)
		r = r * r
		R2 = R2 + r

	R2 = beta2 * R2
	# print("R2 = ", R2)

	return R2

# def L_a(z, proj, kcluster):
# 	loss = 0
# 	for k in range(kcluster):
# 		m = z[k].shape[1]			# z[k] 列向量构成的矩阵，m为向量数即样本数
# 		loss = loss + projection_residual(_z[k], _U[k])/m
#
# 	loss = loss/kcluster
#
# 	return loss
