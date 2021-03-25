import argparse
import os
import sys
import numpy as np
import scipy.io as sio
import tensorflow as tf

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

def parse_arguments(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--save_dir', default="logs/weights.h5")
	parser.add_argument('--epoch', type=int, default=10)
	parser.add_argument('--batch_size', type=int, default=1440)

	return parser.parse_args(argv)

if __name__ == "__main__":
	args = parse_arguments(sys.argv[1:])
	img, label = load_data()
	img, label = utils.shuffle(img, label)
	train_db = tf.data.Dataset.from_tensor_slices((img, img))
	train_db = train_db.batch(1440)
	theta = train(train_db, label, epoch_num=args.epoch, batch_size=args.batch_size)
