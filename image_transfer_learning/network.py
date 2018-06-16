import tensorflow as tf
import numpy as np


class Network:

    def __init__(self, in_dims, num_classes):

        """
        Build the neural network
        """

        # TODO - Figure out which graph
        self.input = tf.placeholder(tf.float32, shape=(None, in_dims))
        logits = tf.layers.dense(self.input, num_classes)

        self.labels = tf.placeholder(tf.int32, shape=None)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits)

        self.learning_rate = tf.placeholder(tf.float32)
        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

    def train(self, input_, labels, num_epochs=10, learning_rate=0.001,
              batch_size=2, verbose=True):

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):

                if verbose:
                    print('Training epoch {}'.format(epoch))

                shuffle_idx = np.arange(input_.shape[0])
                np.random.shuffle(shuffle_idx)

                for idx in range(0, len(shuffle_idx), batch_size):

                    i = shuffle_idx[idx:idx+batch_size]

                    feed_dict = {self.input: input_[i, :],
                                 self.labels: labels[i],
                                 self.learning_rate: learning_rate}

                    _, loss = sess.run([self.opt, self.loss],
                                        feed_dict=feed_dict)

                    print('Loss: {:.2f}'.format(loss[0]))



    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
