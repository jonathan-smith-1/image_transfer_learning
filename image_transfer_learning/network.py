import tensorflow as tf
import numpy as np


class Network:

    def __init__(self, in_dims, num_classes):

        """
        Build the neural network
        """

        tf.reset_default_graph()

        tf.set_random_seed(1234)

        self.input = tf.placeholder(tf.float32, shape=(None, in_dims))
        self.logits = tf.layers.dense(self.input, num_classes)
        self.output = tf.argmax(self.logits, axis=1)

        self.labels = tf.placeholder(tf.int32, shape=None)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=self.logits)

        self.learning_rate = tf.placeholder(tf.float32)
        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def train(self, training_input, training_labels, num_epochs=2,
              learning_rate=0.001, batch_size=2, validate=False,
              validation_input=None, validation_labels=None):

        np.random.seed(42)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(num_epochs):

                print('Training epoch {}'.format(epoch))

                shuffle_idx = np.arange(training_input.shape[0])
                np.random.shuffle(shuffle_idx)

                for idx in range(0, len(shuffle_idx), batch_size):

                    i = shuffle_idx[idx:idx+batch_size]

                    feed_dict = {self.input: training_input[i, :],
                                 self.labels: training_labels[i],
                                 self.learning_rate: learning_rate}

                    _, loss = sess.run([self.opt, self.loss],
                                        feed_dict=feed_dict)

                    print('Loss: {:.2f}'.format(loss[0]))

                if validate and validation_input is not None and \
                        validation_labels is not None:

                    total_results = 0
                    total_correct_results = 0

                    for i in range(0, validation_input.shape[0], batch_size):

                        feed_dict = {self.input: validation_input[i:i+batch_size, :]}

                        out = sess.run(self.output, feed_dict=feed_dict)

                        correct_results = np.equal(out, validation_labels[i:i+batch_size])

                        total_results += correct_results.size
                        total_correct_results += np.sum(correct_results)

                    proportion_correct = total_correct_results/total_results

                    print('Validation accuracy: {:.2f}%'.format(
                        proportion_correct*100))

            save_path = self.saver.save(sess, "./tmp/model.ckpt")
            print("Model saved in path: %s" % save_path)

    def predict(self, feature_vectors, int_to_lab):

        """

        Args:
            feature_vectors: 2D numpy array, any number of rows
            int_to_lab:

        Returns:

        """

        with tf.Session() as sess:

            restore_path = "./tmp/model.ckpt"
            self.saver.restore(sess, restore_path)
            print("Model restored from path: %s" % restore_path)

            pred = sess.run(self.output, feed_dict={self.input:
                                                        feature_vectors})

            return np.array([int_to_lab[i] for i in pred])

    def evaluate(self, test_input, test_labels, batch_size=2):

        total_results = 0
        total_correct_results = 0

        with tf.Session() as sess:

            restore_path = "./tmp/model.ckpt"
            self.saver.restore(sess, restore_path)
            print("Model restored from path: %s" % restore_path)

            for i in range(0, test_input.shape[0], batch_size):
                feed_dict = {self.input: test_input[i:i + batch_size, :]}

                out = sess.run(self.output, feed_dict=feed_dict)

                correct_results = np.equal(out, test_labels[i:i+batch_size])

                total_results += correct_results.size
                total_correct_results += np.sum(correct_results)

            proportion_correct = total_correct_results / total_results

            print('Test accuracy: {:.2f}%'.format(
                proportion_correct * 100))