from image_transfer_learning.network import Network
from image_transfer_learning.image_processing import convert_images, \
    get_feature_vector_size, get_num_classes
import numpy as np


# Configuration
EXTRACT_IMAGES = True
TRAIN_NETWORK = True
EVALUATE_NETWORK = True

# Filepaths
IMAGES_PATH_TRAIN = './data/images/train'
FEATURE_VECTORS_PATH_TRAIN = './data/feature_vectors/train/feature_vectors' \
                             '.pickle'

IMAGES_PATH_VALID = './data/images/valid'
FEATURE_VECTORS_PATH_VALID = './data/feature_vectors/valid/feature_vectors' \
                             '.pickle'

IMAGES_PATH_TEST = './data/images/test'
FEATURE_VECTORS_PATH_TEST = './data/feature_vectors/test/feature_vectors' \
                            '.pickle'

if EXTRACT_IMAGES:

    lab_to_int = convert_images(IMAGES_PATH_TRAIN, FEATURE_VECTORS_PATH_TRAIN)
    convert_images(IMAGES_PATH_TEST, FEATURE_VECTORS_PATH_TEST, lab_to_int)
    convert_images(IMAGES_PATH_VALID, FEATURE_VECTORS_PATH_VALID, lab_to_int)

if TRAIN_NETWORK:

    input_dimension = get_feature_vector_size(FEATURE_VECTORS_PATH_TRAIN)
    num_classes = get_num_classes(FEATURE_VECTORS_PATH_TRAIN)

    net = Network(input_dimension, num_classes)

    training_data = np.load(FEATURE_VECTORS_PATH_TRAIN)
    val_data = np.load(FEATURE_VECTORS_PATH_VALID)

    net.train(training_input=training_data['feature_vectors_array'],
              training_labels=training_data['labels_array'],
              validate=True,
              validation_input=val_data['feature_vectors_array'],
              validation_labels=val_data['labels_array'])

if EVALUATE_NETWORK:

    input_dimension = get_feature_vector_size(FEATURE_VECTORS_PATH_TEST)
    num_classes = get_num_classes(FEATURE_VECTORS_PATH_TEST)

    net = Network(input_dimension, num_classes)

    test_data = np.load(FEATURE_VECTORS_PATH_TEST)

    net.evaluate(test_input=test_data['feature_vectors_array'],
                 test_labels=test_data['labels_array'])
