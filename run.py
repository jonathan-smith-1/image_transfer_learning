"""
Use transfer learning to convert images to feature vectors.  Train a
neural network to classify the images and evaluate the network.
"""

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
TRAIN_IMAGES_PATH = './data/images/train'
TRAIN_FEATURES_PATH = './data/feature_vectors/train/train_data.pickle'

VALID_IMAGES_PATH = './data/images/valid'
VALID_FEATURES_PATH = './data/feature_vectors/valid/valid_data.pickle'

TEST_IMAGES_PATH = './data/images/test'
TEST_FEATURES_PATH = './data/feature_vectors/test/test_data.pickle'

if EXTRACT_IMAGES:

    label_to_int = convert_images(TRAIN_IMAGES_PATH, TRAIN_FEATURES_PATH)
    convert_images(TEST_IMAGES_PATH, TEST_FEATURES_PATH, label_to_int)
    convert_images(VALID_IMAGES_PATH, VALID_FEATURES_PATH, label_to_int)

if TRAIN_NETWORK:

    input_dimension = get_feature_vector_size(TRAIN_FEATURES_PATH)
    num_classes = get_num_classes(TRAIN_FEATURES_PATH)

    net = Network(input_dimension, num_classes)

    training_data = np.load(TRAIN_FEATURES_PATH)
    val_data = np.load(VALID_FEATURES_PATH)

    train_data = {'input': training_data['feature_vectors_array'],
                  'labels': training_data['labels_array']}

    valid_data = {'input': val_data['feature_vectors_array'],
                  'labels': val_data['labels_array']}

    params = {'num_epochs': 2, 'learning_rate': 0.001, 'batch_size': 2}

    net.train(train_data, valid_data, params, save_path="./tmp/model.ckpt")

if EVALUATE_NETWORK:

    input_dimension = get_feature_vector_size(TEST_FEATURES_PATH)
    num_classes = get_num_classes(TEST_FEATURES_PATH)

    net = Network(input_dimension, num_classes)

    test_data = np.load(TEST_FEATURES_PATH)

    net.evaluate(test_input=test_data['feature_vectors_array'],
                 test_labels=test_data['labels_array'],
                 restore_path="./tmp/model.ckpt")

if EXAMPLE_PREDICTION:

    # Example of using this network to make a prediction.
    # Using the first two feature vectors from the test dataset

    input_dimension = get_feature_vector_size(TEST_FEATURES_PATH)
    num_classes = get_num_classes(TEST_FEATURES_PATH)

    net = Network(input_dimension, num_classes)

    test_data = np.load(TEST_FEATURES_PATH)

    label_to_int = test_data['label_to_int']
    int_to_label = {v: k for k, v in label_to_int.items()}

    # Predict the labels of the first two feature vectors in the test data.
    pred_input = test_data['feature_vectors_array'][0:2, :]

    prediction = net.predict(pred_input, restore_path="./tmp/model.ckpt")

    print('Predictions:')
    print([int_to_label[p] for p in prediction])
