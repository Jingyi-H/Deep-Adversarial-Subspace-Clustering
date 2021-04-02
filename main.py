import argparse
import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf

import train_mnist
from train import train
import utils

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

def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

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

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default="coil20")
	parser.add_argument('--save_dir', default="logs/weights.h5")
	parser.add_argument('--epoch', type=int, default=30)
	parser.add_argument('--batch_size', type=int, default=1440)
	parser.add_argument('--pretrain_epoch', type=int, default=50)
	parser.add_argument('--alpha', type=float, default=0.9)
	parser.add_argument('--lr_g', type=float, default=1e-3)
	parser.add_argument('--lr_d', type=float, default=2e-4)
	parser.add_argument('--pretrain', type=bool, default=False)

	return parser.parse_args(argv)

if __name__ == "__main__":
	args = parse_arguments(sys.argv[1:])

	if args.dataset == "mnist":
		x, y = load_mnist()
		train_db = tf.data.Dataset.from_tensor_slices((x, y))
		train_db = train_db.batch(1000).shuffle(30)
		theta = train_mnist.train(train_db, epoch_num=args.epoch, batch_size=args.batch_size, pre_train_epoch=args.pretrain_epoch,
				  alpha=args.alpha, g_lr=args.lr_g, d_lr=args.lr_d, pretrain=args.pretrain)

	elif args.dataset == "coil20":
		x, y = load_data(path="data/COIL20.mat")
		train_db = tf.data.Dataset.from_tensor_slices((x, y))
		train_db = train_db.batch(1440).shuffle(30)

		theta = train(train_db, epoch_num=args.epoch, batch_size=args.batch_size, pre_train_epoch=args.pretrain_epoch,
				  alpha=args.alpha, g_lr=args.lr_g, d_lr=args.lr_d, pretrain=args.pretrain)
