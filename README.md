# image_transfer_learning
Perform multi-class classification of images using transfer learning and a simple classifier.

The aim of this project was to try out transfer learning using [Tensorflow Hub](https://www.tensorflow.org/hub/) on a
complex task. The task I chose was multiclass classification between images of 133 different dog breeds.

<p align="center">
<img src="https://upload.wikimedia.org/wikipedia/commons/6/69/Afghane.jpg">
</p>

The rather extraordinary looking Afghan Hound, presumably one of the easier ones to classify!  Image thanks to 
[Wikipedia Creative Commons CC BY-SA 3.0](https://commons.wikimedia.org/w/index.php?curid=512895)

The code in `run.py` is divided into sections, each of which can be performed individually:
- Square-off and downsample the images, and use transfer learning to produce and save the 'bottleneck' feature vectors.
- Train the model using the feature vectors and the labels.
- Evaluate the model on a withheld test dataset.

The data I used was Udacity's dog dataset [download >1Gb](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip) containing 6680 training, 835 validation and 836 test images of
dogs, spread unevenly across 133 classes. I first started working on this as a piece of coursework while I was taking
Udacity's excellent [Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101). Since
finishing the course I have extended that work into what you see in this repository.

For the transfer learning I used Tensorflow Hub's [Inception V3](https://www.tensorflow.org/hub/modules/google/imagenet/inception_v3/feature_vector/1) network to 
convert
images to feature vectors.  There are loads of other options on Tensorflow 
Hub that easily drop in instead.

The model itself is then a single linear layer (i.e. linear regression!), which produces 85% classification accuracy on
the test data.  More complex models and careful tuning would surely 
outperform this.  Using TensorFlow Hub's [Inception ResNet V2](https://www.tensorflow.org/hub/modules/google/imagenet/inception_resnet_v2/feature_vector/1) model instead 
takes longer to process the images but achieves a slightly higher accuracy 
of 86.1%.

I feel that this  shows the remarkable power of transfer learning, which has the power to turn a complex task such as
this into one that can be rapidly solved with linear regression.



