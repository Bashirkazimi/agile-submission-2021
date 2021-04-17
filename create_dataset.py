from utils import utils
import os
import argparse
from glob import glob


if __name__ == "__main__":

	parser = argparse.ArgumentParser(
		description='Create datasets for trainin, testing and validation!')
	parser.add_argument('--input_tif', required=False,
						default='data/dtms/test_dtm.tif',
						metavar="Large input file to crop training/testing/validation patches from",
						help="Large input file to crop training/testing/validation patches from")
	parser.add_argument('--label_tif', required=False,
						default='data/labels/test_labels.tif',
						metavar="Corresponding large label file to crop training/testing/validation patches from",
						help="Corresponding large label file to crop training/testing/validation patches from")
	parser.add_argument('--data_dir', required=False,
						default='data',
						metavar="directory to save the training/test/validation data",
						help="directory to save the training/test/validation data")
	parser.add_argument('--split', required=False,
						default='test',
						metavar="is it training (train), testing (test) or validation (valid)",
						help="is it training (train), testing (test) or validation (valid)")
	parser.add_argument('--dim', required=False,
						metavar="dimensions of input and output pairs",
						type=int,
						default=128,
						help='dimensions of input and output pairs')
	parser.add_argument('--step', required=False,
						metavar="amount of overlap when creating input/label pairs",
						type=int,
						default=32,
						help='dimensions of input and output pairs')
	# parse arguments!
	args = parser.parse_args()

	# create dataset directory for the corresponding split
	data_dir = args.data_dir
	split = args.split
	data_path = os.path.join(data_dir, split)
	x_dir = os.path.join(data_path, 'x')
	y_dir = os.path.join(data_path, 'y')
	if not os.path.exists(x_dir):
		os.makedirs(x_dir)
	if not os.path.exists(y_dir):
		os.makedirs(y_dir)

	# create and save the dataset
	print('creating dataset!')
	utils.create_dataset(args.input_tif, args.label_tif, args.step, args.dim, x_dir, y_dir)

	# write filenames for input and output in a text file to be used for training/validation/testing
	files = glob(os.path.join(data_path, 'x/*.tif'))
	with open(os.path.join(data_path, 'filenames.txt'), 'w') as writer:
		for file in files:
			pth, filename = os.path.split(file)
			writer.writelines('{}\n'.format(filename))

			




