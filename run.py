from image_transfer_learning.network import Network
from image_transfer_learning.image_processing import convert_images, \
    get_data_shape

# TODO - Make filepaths a bit more robust - currently it assumes the code is run for a certain directory

# convert_images()  # TODO - add filename input

num_features, num_classes = get_data_shape()

net = Network(in_dims=100, num_classes=100)  # TODO - get the right dimensions
#net.train()  # TODO - add data input
#net.predict()  # TODO - add data input.

# TODO - add training etc. of network here



