from image_transfer_learning.network import Network
from image_transfer_learning.image_processing import convert_images, \
    get_feature_vector_size, get_num_classes
import numpy as np


# Configuration
EXTRACT_IMAGES = True
TRAIN_NETWORK = True
EVALUATE_NETWORK = True
EXAMPLE_PREDICTION = True

# Filepaths
IMAGES_PATH_TRAIN = './data/images/train'
FEATURE_VECTORS_PATH_TRAIN = './data/feature_vectors/train/train_data.pickle'

IMAGES_PATH_VALID = './data/images/valid'
FEATURE_VECTORS_PATH_VALID = './data/feature_vectors/valid/valid_data.pickle'

IMAGES_PATH_TEST = './data/images/test'
FEATURE_VECTORS_PATH_TEST = './data/feature_vectors/test/test_data.pickle'

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
              validation_input=val_data['feature_vectors_array'],
              validation_labels=val_data['labels_array'],
              save_path="./tmp/model.ckpt")

if EVALUATE_NETWORK:

    input_dimension = get_feature_vector_size(FEATURE_VECTORS_PATH_TEST)
    num_classes = get_num_classes(FEATURE_VECTORS_PATH_TEST)

    net = Network(input_dimension, num_classes)

    test_data = np.load(FEATURE_VECTORS_PATH_TEST)

    net.evaluate(test_input=test_data['feature_vectors_array'],
                 test_labels=test_data['labels_array'],
                 restore_path="./tmp/model.ckpt")

if EXAMPLE_PREDICTION:

    # Example of using this network to make a prediction
    input_dimension = get_feature_vector_size(FEATURE_VECTORS_PATH_TEST)
    num_classes = get_num_classes(FEATURE_VECTORS_PATH_TEST)

    net = Network(input_dimension, num_classes)

    test_data = np.load(FEATURE_VECTORS_PATH_TEST)

    lab_to_int = test_data['lab_to_int']
    int_to_lab = {v:k for k, v in lab_to_int.items()}

    pred_input = test_data['feature_vectors_array'][0:2, :]  # first two

    prediction = net.predict(pred_input, restore_path="./tmp/model.ckpt")

    print([int_to_lab[p] for p in prediction])
