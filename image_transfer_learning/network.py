import tensorflow as tf


class Network:

    def __init__(self, in_dims, num_classes):

        """
        Build the neural network
        """

        with tf.Graph().as_default():

            input = tf.placeholder(tf.float32, shape=(None, in_dims))
            logits = tf.layers.dense(input, num_classes)

            labels = tf.placeholder(tf.int32, shape=(None))

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)

            lr = tf.placeholder(tf.float32)
            self.opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    def training_step(self, num_epochs, learning_rate, features, labels):

        """

        with tf.Session() as sess:

            # initialise all variables
            for _ in range(num_steps):
                sess.run(self.opt, feed_dict={})
        """


    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
