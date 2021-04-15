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
from sklearn.metrics import classification_report
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
		description='Evaluate HRNet')
	parser.add_argument('--evaluation_file', required=True,
						metavar="evaluation_file",
						help='evaluation_file')
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

	# test_gen = utils.BinaryData(conf.test_dir, test_files, conf.batch_size, False, dim=conf.dim)
	test_gen = utils.MultiData(conf.test_dir, test_files, conf.batch_size, False, dim=conf.dim, num_classes=conf.num_classes)
	
	reset_tf_session()

	model, _, _, phase3 = hrnet.hrnet_segmenter(input_shape=(conf.dim, conf.dim, conf.input_channels), classes=conf.num_classes, halfed=conf.crop_labels, activation=conf.final_activation)

	h5_file = 'files/{}_{}_{}_{}.h5'.format(conf.specific_name, conf.model_type, conf.dim, conf.epochs)
	model.load_weights(h5_file)
	opt = keras.optimizers.Adam(0.001)
	ious = utils.sparse_build_iou_for(list(range(conf.num_classes)), list(range(conf.num_classes)))
	ious.append(utils.sparse_mean_iou)
	metrics = ious

	model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=metrics)
	res = model.evaluate_generator(test_gen, verbose=1)
	print(res)

	result_dict = {'model_type':[conf.model_type], 'specific_name': [conf.specific_name], 'loss': [], 'sparse_iou_0':[],'sparse_iou_1':[],'sparse_iou_2':[],'sparse_iou_3':[],'sparse_iou_4':[],'sparse_iou_5':[],'sparse_mean_iou':[]}
	ks = ['loss', 'sparse_iou_0','sparse_iou_1','sparse_iou_2','sparse_iou_3','sparse_iou_4','sparse_iou_5','sparse_mean_iou'] 
	for e, k in enumerate(ks):
		result_dict[k].append(res[e])

	predictions = np.empty((len(test_files), conf.dim, conf.dim))
	ys = np.empty((len(test_files), conf.dim, conf.dim))
	for e, (x,y) in enumerate(test_gen):
		prediction = model.predict(x)
		predictions[e*conf.batch_size:e*conf.batch_size+conf.batch_size] = np.argmax(prediction, -1)
		ys[e*conf.batch_size:e*conf.batch_size+conf.batch_size] = np.squeeze(y, -1)

	clf = classification_report(ys.reshape(np.prod(ys.shape)), predictions.reshape(np.prod(ys.shape)), labels=[0,1,2,3,4,5], output_dict=True)
	for i in range(conf.num_classes):
		result_dict['prediction_{}'.format(i)] = [clf['{}'.format(i)]['precision']]
		result_dict['recall_{}'.format(i)] = [clf['{}'.format(i)]['recall']]
		result_dict['f1-score_{}'.format(i)] = [clf['{}'.format(i)]['f1-score']]
		result_dict['support_{}'.format(i)] = [clf['{}'.format(i)]['support']]


	result_dict['macro avg precision'] = [clf['macro avg']['precision']]
	result_dict['macro avg recall'] = [clf['macro avg']['recall']]
	result_dict['macro avg f1-score'] = [clf['macro avg']['f1-score']]


	result_dict['weighted avg precision'] = [clf['weighted avg']['precision']]
	result_dict['weighted avg recall'] = [clf['weighted avg']['recall']]
	result_dict['weighted avg f1-score'] = [clf['weighted avg']['f1-score']]

	clf = classification_report(ys.reshape(np.prod(ys.shape)), predictions.reshape(np.prod(ys.shape)), labels=[1,2,3,4,5], output_dict=True)


	result_dict['weighted avg precision ignoring bg'] = [clf['weighted avg']['precision']]
	result_dict['weighted avg recall ignoring bg'] = [clf['weighted avg']['recall']]
	result_dict['weighted avg f1-score ignoring bg'] = [clf['weighted avg']['f1-score']]

	result_dict['micro avg precision'] = [clf['micro avg']['precision']]
	result_dict['micro avg recall'] = [clf['micro avg']['recall']]
	result_dict['micro avg f1-score'] = [clf['micro avg']['f1-score']]


	df = pd.DataFrame(result_dict)
	output_file = 'files/{}'.format(conf.evaluation_file) 
	df.to_csv(output_file, index=False)
	