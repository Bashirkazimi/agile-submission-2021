import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import random 
import tensorflow as tf 

np.random.seed(42)
tf.set_random_seed(42)
random.seed(42)


import config

import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR) 

from models import hrnet, segnet
from utils import utils


import pandas as pd 
import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import logging
import os
import sys
import ogr
import keras
from sklearn.model_selection import train_test_split
import argparse

if __name__ == "__main__":

	conf = config.Config()
	parser = argparse.ArgumentParser(
		description='Make predictions using SegNet')
	parser.add_argument('--output_shp_file', required=True,
						metavar="output file to save the shapes",
						help='Type of input raster')
	parser.add_argument('--input_folder', required=False,
						metavar="Type of input raster",
						help='Type of input raster')
	parser.add_argument('--epochs', required=False,
						metavar="number of epochs",
						type=int,
						help='Number of epochs (default=50)')
	parser.add_argument('--batch_size', required=False,
						metavar="batch size",
						type=int,
						help='Batch size (default=64)')
	parser.add_argument('--dim', required=False,
						metavar="input dimensions",
						type=int,
						help='input dimensions (default=224)')
	parser.add_argument('--model_type', required=False,
						metavar="Type of DL model",
						help='Type of DL model')
	parser.add_argument('--specific_name', required=False,
						metavar="SpecificName for this task",
						help='Specific name for this task')
	parser.add_argument('--pretrained_file', required=False,
						metavar="path to pretrained model feature extractor weights",
						help='path to pretrained model feature extractor weights')
	parser.add_argument('--pretrained', required=False,
						metavar="<True|False>",
						help='pretrained or not',
						type=bool)
	parser.add_argument('--crop_labels', required=False,
						metavar="<True|False>",
						help='crop labels to half or not',
						type=bool)

	args = parser.parse_args()
	for k,v in vars(args).items():
		print(type(v), v)
		if v is not None and hasattr(conf, k):
			setattr(conf, k, v)

	conf.reset_attributes()

	conf.display()

	with open(os.path.join(conf.test_dir, 'filenames.txt'), 'r') as reader:
		test_files = reader.readlines()
		test_files = [f.strip() for f in test_files]

	test_gen = utils.BinaryData(conf.test_dir, test_files, conf.batch_size, False, dim=conf.dim, return_filenames=True)

	keras.backend.clear_session()
	model = segnet.segnetCustomized((conf.dim, conf.dim, conf.input_channels), 1, kernel=3, pool_size=(2, 2), output_mode=conf.final_activation)
	h5_file = 'files/{}_{}_{}_{}.h5'.format(conf.specific_name, conf.model_type, conf.dim, conf.epochs)
	opt = keras.optimizers.Adam(0.001)
	metrics = ['acc', utils.f1_m, utils.precision_m, utils.recall_m]
	model.load_weights(h5_file)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)

	output_shp_file = 'files/{}'.format(conf.output_shp_file)
	utils.make_predictions_and_vectorize_binary(model, test_gen, output_shp_file, conf.test_dir)
	# utils.make_predictions_and_vectorize(model, test_gen, output_shp_file) 
