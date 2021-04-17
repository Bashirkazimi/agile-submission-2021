import keras
import keras.backend as K

import tensorflow as tf 
import numpy as np 
import random

np.random.seed(42)
tf.set_random_seed(42)
random.seed(42)

import gdal, gdalconst

import os


from sklearn.model_selection import train_test_split
import skimage.io
import skimage.transform

import ogr, osr
from glob import glob

def normalize_raster_locally(x):
	mx = x.max()
	mn = x.min()
	x = (x - mn + np.finfo(float).eps) / (mx - mn + np.finfo(float).eps) 
	return x 


def recall_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def precision_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def f1_m(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))


def create_dataset(input_tif, label_tif, step_size, dim, x_dir, y_dir, threshold=0.15):
	ds = gdal.Open(input_tif, gdalconst.GA_ReadOnly)
	xsize = ds.RasterXSize
	ysize = ds.RasterYSize

	for x in range(0, xsize-dim, step_size):
		for y in range(0, ysize-dim, step_size):
			current_label = gdal.Translate('', label_tif, srcWin=[x, y, dim, dim], format='MEM')
			current_array = current_label.ReadAsArray()
			if np.count_nonzero(current_array) >= (threshold*dim*dim):
				print(x, y, xsize, ysize, np.unique(current_array, return_counts=True))
				gdal.Translate(os.path.join(x_dir,'{}_{}.tif').format(x, y), input_tif, srcWin=[x, y, dim, dim])
				gdal.Translate(os.path.join(y_dir,'{}_{}.tif').format(x, y), label_tif, srcWin=[x, y, dim, dim])



def get_input_label_pairs(input_dir, filenames, dim, multi_class=False):
	batch_size = len(filenames)
	xs = np.empty((len(filenames), dim, dim))
	ys = np.empty((len(filenames), dim, dim))
	for e, file in enumerate(filenames):
		im = skimage.io.imread(os.path.join(input_dir, 'x', file), plugin='pil')
		label = skimage.io.imread(os.path.join(input_dir, 'y', file), plugin='pil')
		if not multi_class:
			label[label != 0] = 1
		xs[e] = normalize_raster_locally(im)
		ys[e] = label 
	return np.expand_dims(xs, -1), np.expand_dims(ys, -1)


class BinaryData(keras.utils.Sequence):
	def __init__(self, input_dir, filenames, batch_size, shuffle, dim=128, return_filenames=False):
		self.input_dir = input_dir
		self.filenames = filenames
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.dim = dim 
		self.return_filenames = return_filenames
		self.on_epoch_end()
	
	def __len__(self):
		return int(np.ceil(len(self.filenames) / self.batch_size))
	
	def __getitem__(self, index):
		filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
		x, y = get_input_label_pairs(self.input_dir, filenames, self.dim)
		if self.return_filenames:
			return x, y, filenames
		else:
			return x, y

	def on_epoch_end(self):
		if self.shuffle == True:
			np.random.shuffle(self.filenames)


class MultiData(keras.utils.Sequence):
	def __init__(self, input_dir, filenames, batch_size, shuffle, dim=128, return_filenames=False, num_classes=6):
		self.input_dir = input_dir
		self.filenames = filenames
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.dim = dim 
		self.return_filenames = return_filenames
		self.num_classes = 6
		self.on_epoch_end()
	
	def __len__(self):
		return int(np.ceil(len(self.filenames) / self.batch_size))
	
	def __getitem__(self, index):
		filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
		x, y = get_input_label_pairs(self.input_dir, filenames, self.dim, multi_class=True)
		if self.return_filenames:
			return x, y, filenames
		else:
			return x, y

	def on_epoch_end(self):
		if self.shuffle == True:
			np.random.shuffle(self.filenames)


def mask_to_polygon_in_memory(original_data, array, field_name='label'):
	geotrans = original_data.GetGeoTransform()
	proj = original_data.GetProjection()

	driver = gdal.GetDriverByName('MEM')
	dataset = driver.Create('', array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
	dataset.GetRasterBand(1).WriteArray(array)
	dataset.SetProjection(proj)
	dataset.SetGeoTransform(geotrans)
	band = dataset.GetRasterBand(1)

	driver_mask = gdal.GetDriverByName('MEM')
	ds_mask = driver_mask.Create('', array.shape[1], array.shape[0], 1, gdal.GDT_Float32)
	ds_mask.SetGeoTransform(geotrans)
	ds_mask.SetProjection(proj)
	ds_mask_array = (array>0).astype(np.int32)
	ds_mask.GetRasterBand(1).WriteArray( ds_mask_array )
	mask_band = ds_mask.GetRasterBand(1)

	srs = osr.SpatialReference(wkt=proj)
	driver = gdal.GetDriverByName("Memory")
	outDatasource = driver.Create('',0,0,0,gdal.GDT_Float32)
	# outDatasource = driver.CreateDataSource('')
	outLayer = outDatasource.CreateLayer("polygonized", srs=srs)
	if field_name is None:
		field_name='MyFLD'
	newField = ogr.FieldDefn(field_name, ogr.OFTInteger)
	outLayer.CreateField(newField)

	gdal.Polygonize(band, mask_band, outLayer, 0, [], callback=None )
	return outDatasource


def merge_shp_files_from_memory(shp_files, outputMergefn):
	driverName = 'ESRI Shapefile'
	geometryType = ogr.wkbPolygon
	out_driver = ogr.GetDriverByName( driverName )
	if os.path.exists(outputMergefn):
		out_driver.DeleteDataSource(outputMergefn)
	out_ds = out_driver.CreateDataSource(outputMergefn)
	out_layer = out_ds.CreateLayer(outputMergefn, geom_type=geometryType)
	only_ones = True
	for ds in shp_files:
		# ds = ogr.Open(file)
		lyr = ds.GetLayer()
		if only_ones:
			lyr_def = lyr.GetLayerDefn ()
			for i in range(lyr_def.GetFieldCount()):
				out_layer.CreateField (lyr_def.GetFieldDefn(i) )
			only_ones=False
		for feat in lyr:
			out_layer.CreateFeature(feat)
		del ds, lyr
	del out_ds, out_layer
	

def make_predictions_and_vectorize_binary(model, test_data, output_shp_file, input_dir, field_name='label'):
	list_of_shps = []
	batch_counter = 0

	for x, y, files in test_data:
		predictions = model.predict(x)
		predictions = (predictions > 0.5).astype(np.uint8)
		for i in range(len(files)):
			cur_pred = predictions[i]
			cur_pred = np.squeeze(cur_pred, -1)
			original_data = gdal.Open(os.path.join(input_dir, 'x', files[i]), gdalconst.GA_ReadOnly)
			cur_shp = mask_to_polygon_in_memory(original_data, cur_pred, field_name=field_name)
			list_of_shps.append(cur_shp)
		if batch_counter % 10 == 0:
			print('done with batch: {}'.format(batch_counter))
		batch_counter += 1

	print('merging shp data!')
	merge_shp_files_from_memory(list_of_shps, output_shp_file)


def make_predictions_and_vectorize_multiclass(model, test_data, output_shp_file, input_dir, field_name='label'):
	list_of_shps = []
	batch_counter = 0

	for x, y, files in test_data:
		predictions = model.predict(x)
		predictions = np.argmax(predictions, -1)
		for i in range(len(files)):
			cur_pred = predictions[i]
			# cur_pred = np.squeeze(cur_pred, -1)
			original_data = gdal.Open(os.path.join(input_dir, 'x', files[i]), gdalconst.GA_ReadOnly)
			cur_shp = mask_to_polygon_in_memory(original_data, cur_pred, field_name=field_name)
			list_of_shps.append(cur_shp)
		if batch_counter % 10 == 0:
			print('done with batch: {}'.format(batch_counter))
		batch_counter += 1

	print('merging shp data!')
	merge_shp_files_from_memory(list_of_shps, output_shp_file)


def make_predictions_binary(model, tif_file, dim, output_shp_file, field_name='label', batch_size=32, step=None):
	if step is None:
		step = dim
	list_of_tiles = []
	list_of_arrays = []
	list_of_shps = []
	
	ds = gdal.Open(tif_file, gdalconst.GA_ReadOnly)
	xs = ds.RasterXSize
	ys = ds.RasterYSize

	batch_counter = 0

	print('Reading and making predictions!')
	for i in range(0, xs, step):
		for j in range(0, ys, step):
			current_window = gdal.Translate('', tif_file, srcWin=[i, j, dim, dim], format='MEM')#srcWin=[dim*i, dim*j, dim, dim], format='MEM')
			list_of_tiles.append(current_window)
			cur_array = normalize_raster_locally(current_window.ReadAsArray())
			list_of_arrays.append(cur_array)
			batch_counter += 1
			if batch_counter == batch_size:
				print(batch_counter, len(list_of_arrays), len(list_of_tiles), len(list_of_shps))
				inputs = np.array(list_of_arrays)
				inputs = np.expand_dims(inputs, -1)
				prediction = model.predict(inputs, batch_size=batch_size)
				prediction = (prediction > 0.5).astype(np.uint8)
				prediction = np.squeeze(prediction, -1)
				for example in range(batch_size):
					cur_pred = prediction[example]
					original_data = list_of_tiles[example]
					cur_shp = mask_to_polygon_in_memory(original_data, cur_pred, field_name=field_name)
					list_of_shps.append(cur_shp)

				list_of_tiles = []
				list_of_arrays = []

				batch_counter = 0
	
	if batch_counter:
		print('\n\n batch counter: {}\n\n'.format(batch_counter))
		print(batch_counter, len(list_of_arrays), len(list_of_tiles), len(list_of_shps))
		prediction = model.predict(np.expand_dims(np.array(list_of_arrays), -1))
		prediction = (prediction > 0.5).astype(np.uint8)
		prediction = np.squeeze(prediction, -1)
		for example in range(prediction.shape[0]):
			cur_pred = prediction[example]
			original_data = list_of_tiles[example]
			cur_shp = mask_to_polygon_in_memory(original_data, cur_pred, field_name=field_name)
			list_of_shps.append(cur_shp)

	print('merging shp data!')
	merge_shp_files_from_memory(list_of_shps, output_shp_file)


def make_predictions_multiclass(model, tif_file, dim, output_shp_file, field_name='label', batch_size=32, step=None):
    if step is None:
        step = dim
    list_of_tiles = []
    list_of_arrays = []
    list_of_shps = []
    
    ds = gdal.Open(tif_file, gdalconst.GA_ReadOnly)
    xs = ds.RasterXSize
    ys = ds.RasterYSize

    batch_counter = 0

    print('Reading and making predictions!')
    for i in range(0, xs, step):
        for j in range(0, ys, step):
            current_window = gdal.Translate('', tif_file, srcWin=[i, j, dim, dim], format='MEM')#srcWin=[dim*i, dim*j, dim, dim], format='MEM')
            list_of_tiles.append(current_window)
            cur_array = normalize_raster_locally(current_window.ReadAsArray())
            list_of_arrays.append(cur_array)
            batch_counter += 1
            if batch_counter == batch_size:
                print(batch_counter, len(list_of_arrays), len(list_of_tiles), len(list_of_shps))
                inputs = np.array(list_of_arrays)
                inputs = np.expand_dims(inputs, -1)
                predictions = model.predict(inputs, batch_size=batch_size)
                predictions = np.argmax(predictions, -1)
                for example in range(batch_size):
                    cur_pred = predictions[example]
                    original_data = list_of_tiles[example]
                    cur_shp = mask_to_polygon_in_memory(original_data, cur_pred, field_name=field_name)
                    list_of_shps.append(cur_shp)

                list_of_tiles = []
                list_of_arrays = []

                batch_counter = 0
    
    if batch_counter:
        print('\n\n batch counter: {}\n\n'.format(batch_counter))
        print(batch_counter, len(list_of_arrays), len(list_of_tiles), len(list_of_shps))
        predictions = model.predict(np.expand_dims(np.array(list_of_arrays), -1))
        predictions = np.argmax(predictions, -1)
        for example in range(predictions.shape[0]):
            cur_pred = predictions[example]
            original_data = list_of_tiles[example]
            cur_shp = mask_to_polygon_in_memory(original_data, cur_pred, field_name=field_name)
            list_of_shps.append(cur_shp)

    print('merging shp data!')
    merge_shp_files_from_memory(list_of_shps, output_shp_file)	



def sparse_iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(y_true, label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    y_true = K.squeeze(y_true, axis=-1)
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def sparse_mean_iou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1]
    # initialize a variable to store total IoU in
    total_iou = K.variable(0)
    # iterate over labels to calculate IoU for
    for label in range(num_labels):
        total_iou = total_iou + sparse_iou(y_true, y_pred, label)
    # divide total IoU by number of labels to get mean IoU
    return total_iou / num_labels


def sparse_build_iou_for(label: int, name: str = None):
    """
    Build an Intersection over Union (IoU) metric for a label.
    Args:
        label: the label to build the IoU metric for
        name: an optional name for debugging the built method
    Returns:
        a keras metric to evaluate IoU for the given label

    Note:
        label and name support list inputs for multiple labels
    """
    # handle recursive inputs (e.g. a list of labels and names)
    if isinstance(label, list):
        if isinstance(name, list):
            return [sparse_build_iou_for(l, n) for (l, n) in zip(label, name)]
        return [sparse_build_iou_for(l) for l in label]

    # build the method for returning the IoU of the given label
    def label_iou(y_true, y_pred):
        """
        Return the Intersection over Union (IoU) score for {0}.
        Args:
            y_true: the expected y values as a one-hot
            y_pred: the predicted y values as a one-hot or softmax output
        Returns:
            the scalar IoU value for the given label ({0})
        """.format(label)
        return sparse_iou(y_true, y_pred, label)

    # if no name is provided, us the label
    if name is None:
        name = label
    # change the name of the method for debugging
    label_iou.__name__ = 'sparse_iou_{}'.format(name)

    return label_iou