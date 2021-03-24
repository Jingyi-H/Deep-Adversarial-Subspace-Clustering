import numpy as np
import tensorflow as tf
from munkres import Munkres

def shuffle(x, y):
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index, :, :, :]
    y = y[index, :]

    return x, y

def u_regularize(U):
	d = U.shape[0]
	col_norm = tf.norm(U, axis=0, keepdims=True)
	coef = tf.tile(col_norm, tf.constant([d,1]))
	U = U/coef

	return U


def best_map(L1, L2):
	#L1 should be the real labels and L2 should be the clustering number we got
	Label1 = np.unique(L1)       # 去除重复的元素，由小大大排列
	nClass1 = len(Label1)        # 标签的大小
	Label2 = np.unique(L2)
	nClass2 = len(Label2)
	nClass = np.maximum(nClass1,nClass2)
	G = np.zeros((nClass, nClass))
	for i in range(nClass1):
		ind_cla1 = L1 == Label1[i]
		# ind_cla1 = ind_cla1.astype(float)
		for j in range(nClass2):
			ind_cla2 = L2 == Label2[j]
			# ind_cla2 = ind_cla2.astype(float)
			G[i,j] = np.sum(ind_cla2 * ind_cla1)
	m = Munkres()
	index = m.compute(-G.T)
	index = np.array(index)
	c = index[:,1]
	newL2 = np.zeros(L2.shape)
	for i in range(nClass2):
		newL2[L2 == Label2[i]] = Label1[c[i]]

	return newL2

def generate_data(z_k, m_k, m_gen):
	'''
	generate noise in uniform distribution
	:param m: 生成m个样本
	:return: alpha - coefficience matrix of representation z
	'''
	# z_k = z_k.T

	for i in range(m_gen):
		alpha = np.random.random(m_k).reshape(-1, 1)
		alpha = tf.cast(alpha, dtype='float64')
		z_k = tf.cast(z_k, dtype='float64')
		_z = tf.matmul(z_k, alpha)
		# _z = z_k.dot(alpha)
		# _z = tf.reshape(-1, 1)
		if i == 0:
			gen = _z
		else:
			gen = np.hstack([gen, _z])
	# print(gen.shape)

	return gen

def random_select(labels):
	pass

def qr_decomp(matrix):
	pass
