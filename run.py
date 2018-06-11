import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from image_transfer_learning.functions import make_square
from skimage.transform import resize
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

image_path = "./dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg"
image = imread(image_path)
image = make_square(image)

with tf.Graph().as_default():

    module = hub.Module("https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1")
    height, width = hub.get_expected_image_size(module)
    batch_size = 1
    images = tf.placeholder(tf.float32, shape=[batch_size, height, width, 3], name='Input_images')

    features = module(images)  # Features with shape [batch_size, num_features].

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        input_image = resize(image, (height, width), anti_aliasing=True)
        input_image = np.expand_dims(input_image, axis=0)

        feature_vector = sess.run(features, feed_dict={images: input_image})

        debug = 0




"""
images = []
labels = []

m = hub.Module("path/to/a/module_dir")

#for category_dir in os.scandir("./dogImages/train"):

#    label = os.path.basename(os.path.normpath(category_dir))

for image_path in os.scandir(category_dir):

    images.append(make_square(imread(image_path.path)))




image_path = "./dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg"
img = imread(image_path)
image = make_square(img)

imshow(image)
plt.show()

"""
