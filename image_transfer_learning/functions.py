from skimage.io import imread
from skimage.transform import resize
import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from image_transfer_learning.image_processing import make_square


def convert_images():

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

			for dataset in ['train', 'test', 'valid']:

				for category_dir in os.scandir("./images/" + dataset):

					label = os.path.basename(os.path.normpath(category_dir))

					for image_path in os.scandir(category_dir):
						print('Converting image: ' + image_path.name)
						image = imread(os.path.abspath(image_path))
						image = make_square(image)

						# Constant argument prevents deprication warning
						image = resize(image, (height, width), anti_aliasing=True, mode='constant')
						image = np.expand_dims(image, axis=0)

						vec = sess.run(features, feed_dict={images: image})

						feature_vectors.append(vec)
						labels.append(label)

				feature_vectors_array = np.concatenate(feature_vectors, axis=0)
				labels_array = np.array(labels)

				# save the arrays as a pickle file here
				feature_vectors_save_path = "./feature_vectors/" + dataset
				if not os.path.exists(feature_vectors_save_path):
					os.makedirs(feature_vectors_save_path)

				np.savez(feature_vectors_save_path + '/vectors.npz', feature_vectors_array=feature_vectors_array,
						 labels_array=labels_array)