from skimage.io import imread
from skimage.transform import resize
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def make_square(img):
    """
    Trim an image to make it square by keeping the centre of the original image.

    Args:
        img (Numpy array): Input image with shape (height, width, channels)

    Returns:
        Numpy array of trimmed image.

    """
    height, width, channels = img.shape

    if height >= width:
        h_max = int(height/2 + width/2)
        h_min = int(height/2 - width/2)

        return img[h_min:h_max, :, :].copy()

    else:
        w_max = int(width/2 + height/2)
        w_min = int(width/2 - height/2)

        return img[:, w_min:w_max, :].copy()


def convert_images(images_path, feature_vectors_path, lab_to_int):
    """
    Convert images into feature vectors.

    Expecting a file structure...

    Saves them in this format...

    Args:
        images_path (string): Filepath to folder
        feature_vectors_path (string): Filepath to folder containing feature vectors.
        lab_to_int (dict): Mapping from labels to integers

    Returns:
        Nothing

    """
    print('Converting images from: ' + images_path)

    # Convert the images to vectors and to a numpy array
    feature_vectors = []
    labels = []

    with tf.Graph().as_default():

        mod = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
        height, width = hub.get_expected_image_size(mod)
        batch_size = 1
        images = tf.placeholder(tf.float32, shape=[batch_size, height, width, 3], name='Input_images')

        features = mod(images)  # Features with shape [batch_size, num_features].

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            for category_dir in os.scandir(images_path):

                label = os.path.basename(os.path.normpath(category_dir))

                for image_path in os.scandir(category_dir):
                    print(image_path.name)
                    image = imread(os.path.abspath(image_path))
                    image = make_square(image)

                    # Constant argument prevents deprication warning
                    image = resize(image, (height, width), anti_aliasing=True,
                                   mode='constant')
                    image = np.expand_dims(image, axis=0)

                    vec = sess.run(features, feed_dict={images: image})

                    feature_vectors.append(vec)
                    labels.append(lab_to_int[label])

            feature_vectors_array = np.concatenate(feature_vectors, axis=0)
            labels_array = np.array(labels)

            np.savez(feature_vectors_path,
                     feature_vectors_array=feature_vectors_array,
                     labels_array=labels_array)


def get_feature_vector_size(feature_vector_path):

    data = np.load(feature_vector_path)

    return data['feature_vectors_array'].shape[1]


def get_num_classes(feature_vector_path):

    data = np.load(feature_vector_path)

    unique_labels = np.unique(data['labels_array'])
    return unique_labels.size


def enumerate_labels(images_path):

    labels = set()

    for category_dir in os.scandir(images_path):

        labels.add(os.path.basename(os.path.normpath(category_dir)))

    int_to_lab = dict(enumerate(sorted(labels)))
    lab_to_int = {v: k for k, v in int_to_lab.items()}

    return int_to_lab, lab_to_int





