import numpy as np
from sklearn.cluster import SpectralClustering
import utils

def get_acc(theta, label):
	label = label.astype(int)
	affinity = np.abs(theta + theta.T)
	affinity = np.abs(affinity)
	sc = SpectralClustering(20, affinity='precomputed', n_init=100, assign_labels='discretize')
	pred = sc.fit_predict(affinity)
	pred = pred.astype(int)
	label = label.reshape(pred.shape)
	pred = utils.best_map(label, pred)

	num = np.argwhere(pred-label==0).shape[0]

	return num/label.shape[0]
