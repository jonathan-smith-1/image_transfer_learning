from image_transfer_learning.network import Network
from image_transfer_learning.image_processing import convert_images, \
    get_feature_vector_size, get_num_classes, enumerate_labels
import numpy as np


# Configuration
EXTRACT_IMAGES = False
TRAIN_NETWORK = False
EVALUATE_NETWORK = True

# Filepaths
IMAGES_PATH_TRAIN = './data/images/train'
FEATURE_VECTORS_PATH_TRAIN = './data/feature_vectors/train/feature_vectors.npz'

IMAGES_PATH_VALID = './data/images/valid'
FEATURE_VECTORS_PATH_VALID = './data/feature_vectors/valid/feature_vectors.npz'

IMAGES_PATH_TEST = './data/images/test'
FEATURE_VECTORS_PATH_TEST = './data/feature_vectors/test/feature_vectors.npz'

int_to_lab, lab_to_int = enumerate_labels(IMAGES_PATH_TRAIN)

if EXTRACT_IMAGES:

    convert_images(IMAGES_PATH_TRAIN, FEATURE_VECTORS_PATH_TRAIN, lab_to_int)
    convert_images(IMAGES_PATH_TEST, FEATURE_VECTORS_PATH_TEST, lab_to_int)
    convert_images(IMAGES_PATH_VALID, FEATURE_VECTORS_PATH_VALID, lab_to_int)

if TRAIN_NETWORK:
    input_dimension = get_feature_vector_size(FEATURE_VECTORS_PATH_TRAIN)
    num_classes = get_num_classes(FEATURE_VECTORS_PATH_TRAIN)

    net = Network(input_dimension, num_classes)

    training_data = np.load(FEATURE_VECTORS_PATH_TRAIN)
    val_data = np.load(FEATURE_VECTORS_PATH_VALID)

    net.train(training_data['feature_vectors_array'], training_data['labels_array'],
              validate=True, validation_input=val_data['feature_vectors_array'],
              validation_labels=val_data['labels_array'])

if EVALUATE_NETWORK:
    input_dimension = get_feature_vector_size(FEATURE_VECTORS_PATH_TRAIN)
    num_classes = get_num_classes(FEATURE_VECTORS_PATH_TRAIN)

    net = Network(input_dimension, num_classes)

    test_data = np.load(FEATURE_VECTORS_PATH_TEST)

    net.evaluate(test_data['feature_vectors_array'], test_data['labels_array'])


# TODO - Add random seeds






