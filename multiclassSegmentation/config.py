

import numpy as np


class Config(object):

    input_dir = r'../data'
    train_dir = r'../data/train'
    test_dir = r'../data/test'
    valid_dir = r'../data/valid'
    filenames = 'filenames.txt'


    model_type = 'hrnet' 
    dim = 128
    epochs = 50
    specific_name = 'hrnetMulti'#'iterative_binary'
    batch_size = 20
    input_channels = 1
    num_classes = 6
    final_activation = 'softmax'
    crop_labels = False
    normalized = True
    normalization_type = 'local'
    pretrained = False
    pretrained_file = 'files/pth.h5'
    evaluation_file = 'evaluation.csv'
    confidence_threshold = 0.80
    keep_percent = 0.1 * dim * dim
    iterations = 50
    output_shp_file = 'hrnetMulti.shp'
    input_dtm = r'../data/dtms/test_dtm.tif'



    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.model_path = 'files/{}_{}_{}_{}.h5'.format(self.specific_name,self.model_type, self.dim, self.epochs)
        self.hist_path = 'files/{}_{}_{}_{}.csv'.format(self.specific_name,self.model_type, self.dim, self.epochs)

    def reset_attributes(self):
        self.model_path = 'files/{}_{}_{}_{}.h5'.format(self.specific_name, self.model_type, self.dim, self.epochs)
        self.hist_path = 'files/{}_{}_{}_{}.csv'.format(self.specific_name, self.model_type, self.dim, self.epochs)


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
