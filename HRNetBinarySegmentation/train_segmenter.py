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

from models import hrnet
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

import keras.backend as K



def reset_tf_session():
    curr_session = tf.get_default_session()
    # close current session
    if curr_session is not None:
        curr_session.close()
    # reset graph
    K.clear_session()
    # create new session
    config = tf.ConfigProto(device_count = {'GPU': 1 , 'CPU': 4} )
    # config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    s = tf.InteractiveSession(config=config)
    K.set_session(s)
    return s

if __name__ == "__main__":

	conf = config.Config()
	parser = argparse.ArgumentParser(
		description='Train HRNet')

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

	
	input_dir = conf.input_dir
	train_dir = conf.train_dir
	test_dir = conf.test_dir
	valid_dir = conf.valid_dir

	args = parser.parse_args()
	for k,v in vars(args).items():
		print(type(v), v)
		if v is not None and hasattr(conf, k):
			setattr(conf, k, v)

	conf.reset_attributes()

	conf.display()

	with open(os.path.join(train_dir, 'filenames.txt'), 'r') as reader:
		train_files = reader.readlines()
		train_files = [f.strip() for f in train_files]

	with open(os.path.join(valid_dir, 'filenames.txt'), 'r') as reader:
		valid_files = reader.readlines()
		valid_files = [f.strip() for f in valid_files]

	train_gen = utils.BinaryData(conf.train_dir, train_files, conf.batch_size, True, dim=conf.dim)
	valid_gen = utils.BinaryData(conf.valid_dir, valid_files, conf.batch_size, False, dim=conf.dim)

	reset_tf_session()
	
	model, _, _, phase3 = hrnet.hrnet_segmenter(input_shape=(conf.dim, conf.dim, conf.input_channels), classes=conf.num_classes, halfed=conf.crop_labels, activation=conf.final_activation)

	if conf.pretrained:
		print('loading weight files from {}'.format(conf.pretrained_file))
		phase3.load_weights(conf.pretrained_file)

	opt = keras.optimizers.Adam(0.001)
	# ious = utils.sparse_build_iou_for(list(range(conf.num_classes)), list(range(conf.num_classes)))
	# ious.append(utils.sparse_mean_iou)
	metrics = ['acc', utils.f1_m, utils.precision_m, utils.recall_m ]

	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=metrics)

	checkpoint = ModelCheckpoint(conf.model_path, monitor='val_f1_m', mode='max', save_best_only=True, verbose=1)
	reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=5, min_lr=0.0000001, verbose=1)
	early = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)

	callbacks = [checkpoint, reduce_lr, early]

	hist = model.fit_generator(
		train_gen,
		epochs=conf.epochs,
		validation_data=valid_gen,
		callbacks=callbacks,
		verbose=1
	)

	df = pd.DataFrame(hist.history) 
	df.to_csv(conf.hist_path, index=False) 



