from image_transfer_learning.network import Network
from image_transfer_learning.image_processing import convert_images, \
    get_feature_vector_size, get_num_classes, enumerate_labels
import numpy as np


# Configuration
EXTRACT_IMAGES = False
TRAIN_NETWORK = True

# Filepaths
IMAGES_PATH_TRAIN = './data/images/train'
FEATURE_VECTORS_PATH_TRAIN = './data/feature_vectors/train/feature_vectors.npz'

IMAGES_PATH_TEST = './data/images/test'
FEATURE_VECTORS_PATH_TEST = './data/feature_vectors/test/feature_vectors.npz'


#if EXTRACT_IMAGES:
int_to_lab, lab_to_int = enumerate_labels(IMAGES_PATH_TRAIN)

convert_images(IMAGES_PATH_TRAIN, FEATURE_VECTORS_PATH_TRAIN, lab_to_int)
convert_images(IMAGES_PATH_TEST, FEATURE_VECTORS_PATH_TEST, lab_to_int)

#if TRAIN_NETWORK:
input_dimension = get_feature_vector_size(FEATURE_VECTORS_PATH_TRAIN)
num_classes = get_num_classes(FEATURE_VECTORS_PATH_TRAIN)

net = Network(input_dimension, num_classes)

data = np.load(FEATURE_VECTORS_PATH_TRAIN)
net.train(data['feature_vectors_array'], data['labels_array'])


test_data = np.load(FEATURE_VECTORS_PATH_TEST)

#net.predict(test_data['feature_vectors_array'][0, None], int_to_lab)

#net.evaluate()

# TODO - Add save and restore methods or arguments
# TODO - Make a predict method, which takes in an actual image path
# TODO - Make an evaluate method, which evaluates the images in a given .npz
#  file, could be training or test.
# TODO - Add validation loss to training, after every epoch
# TODO - Convert int_to_lab, lab_to_int to being calculated from the label
# vectors




