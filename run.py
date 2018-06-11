import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from image_transfer_learning.functions import make_square


image_path = "./dogImages/train/001.Affenpinscher/Affenpinscher_00001.jpg"
img = imread(image_path)
image = make_square(img)

imshow(image)
plt.show()
