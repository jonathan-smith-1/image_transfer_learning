"""Image processing functions."""

import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from skimage.io import imread
from skimage.transform import resize
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Needed for large images


def make_square(img):
    """
    Trim an image to make it square by keeping the centre part.

    Args:
        img (Numpy array): Input image with shape (height, width, channels)

    Returns:
        Numpy array of trimmed image, with shape (new_height, new_width,
        channels)

    """
    height, width, _ = img.shape

    if height >= width:
        h_max = int(height/2 + width/2)
        h_min = int(height/2 - width/2)

        trimmed_image = img[h_min:h_max, :, :].copy()

    else:
        w_max = int(width/2 + height/2)
        w_min = int(width/2 - height/2)

        trimmed_image = img[:, w_min:w_max, :].copy()

    return trimmed_image


def convert_images(images_path, save_path, lab_to_int=None):
    """
    Convert images into feature vectors and saves them in a pickle file.

    This function uses transfer learning.  A pre-trained network is loaded
    and used.

    A dictionary mapping labels to integers can be passed in, or can be
    generated and returned.  This is so it can be reused on other datasets.
    E.g. the training data may have more classes in than the test data,
    so this mapping needs to be created using the training data and then
    reused on the validation and test data.

    Args:
        images_path (string): Filepath of the directory containing the
                              training images.  The images must be in
                              folders with the category names.

                              A suitable file structure is shown below:

                              |- images_path/
                              |    |- category_1
                              |         |- image_1.jpg
                              |         |- image_2.jpg
                              |         |- ...
                              |    |- category_2
                              |         |- image_3.jpg
                              |         |- image_4.jpg
                              |         |- ...
                              |    |- ...

        save_path (string): Filepath to a pickle file that will be created
                            by this function.

        lab_to_int (dict): Mapping from labels (strings) to integers.
                           Optional argument.  If provided, this dictionary
                           will be used.  If not provided, then this
                           dictionary will be generated.

    Returns:
        A dictionary mapping from labels (strings) to integers.

    """
    print('Converting images from: ' + images_path)

    # Convert each image to a feature vector
    feature_vectors = []
    labels = []

    if not lab_to_int:
        _, lab_to_int = enumerate_labels(images_path)

    with tf.Graph().as_default():

        mod = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/"
                         "feature_vector/1")
        height, width = hub.get_expected_image_size(mod)

        # [batch_size, height, width, channels]
        images = tf.placeholder(tf.float32,
                                shape=[1, height, width, 3],
                                name='Input_images')

        # Features have shape [batch_size, num_features].
        features = mod(images)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for category_dir in os.scandir(images_path):

                label = os.path.basename(os.path.normpath(category_dir))

                for image_path in os.scandir(category_dir):
                    print(image_path.name)
                    image = imread(os.path.abspath(image_path))
                    image = make_square(image)

                    # Constant argument prevents deprecation warning
                    image = resize(image, (height, width), anti_aliasing=True,
                                   mode='constant')
                    image = np.expand_dims(image, axis=0)

                    vec = sess.run(features, feed_dict={images: image})

                    feature_vectors.append(vec)
                    labels.append(lab_to_int[label])

            feature_vectors_array = np.concatenate(feature_vectors, axis=0)
            labels_array = np.array(labels)

            data = {'feature_vectors_array': feature_vectors_array,
                    'labels_array': labels_array,
                    'label_to_int': lab_to_int}

            with open(save_path, 'wb') as file:
                pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    return lab_to_int


def get_feature_vector_size(path):
    """
    Get the length of the feature vectors.

    Feature vectors are assumed to be in a pickle file or npz file (or
    similar) that loads into a dictionary with key, value pair of
    'feature_vectors_array' and a 2D numpy array of feature vectors.  The
    feature vectors array is a 2D numpy array of shape (num_vectors,
    vector_length).

    Args:
        path (string): Path of file containing feature vectors.

    Returns:
        Nothing

    """
    data = np.load(path)
    return data['feature_vectors_array'].shape[1]


def get_num_classes(path):
    """
    Get the number of classes in the data.

    Together with the feature vectors and the labels is a dictionary mapping
    the labels to integers.  The size of this dictionary gives the number
    of classes.

    The labels to integers dictionary is assumed to be in a pickle file or
    npz file (or similar) that loads into a dictionary with key, value pair of
    'label_to_int' and this dictionary.

    Args:
        path (string): Path of file containing the mapping of labels to
                       integers.

    Returns:
        Nothing

    """
    data = np.load(path)
    return len(data['label_to_int'])


def enumerate_labels(path):
    """
    Create dictionaries mapping label to integer and integer to label.

    Args:
        path (string): Filepath of the directory folders named after each
                       category.

                       A suitable file structure is shown below:

                       |- images_path/
                       |    |- category_1
                       |         |- image_1.jpg
                       |         |- image_2.jpg
                       |         |- ...
                       |    |- category_2
                       |         |- image_3.jpg
                       |         |- image_4.jpg
                       |         |- ...
                       |    |- ...

    Returns:
        Nothing

    """
    labels = set()

    for category_dir in os.scandir(path):

        labels.add(os.path.basename(os.path.normpath(category_dir)))

    int_to_lab = dict(enumerate(sorted(labels)))
    lab_to_int = {v: k for k, v in int_to_lab.items()}

    return int_to_lab, lab_to_int
