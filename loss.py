import tensorflow as tf

def projection_residual(z):
	pass

def reconst_loss(x_true, x_reconst):
	return tf.reduce_mean(tf.square(x_true - x_reconst))

def ae_loss(x_true, x_reconst, z_conv, z_se, theta, lambda1=1, lambda2=1, lambda3=1):
	reconst_loss = tf.reduce_mean(tf.square(x_true - x_reconst))
	self_expr_loss = tf.reduce_mean(tf.square(z_se - z_conv))
	penalty = (tf.norm(theta))^2

	return lambda1 * reconst_loss + lambda2 * self_expr_loss + lambda3 * penalty
