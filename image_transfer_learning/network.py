"""Neural network for image classification using transfer learning."""

import tensorflow as tf
import numpy as np


class Network:
    """Neural network for multi-class classification."""

    def __init__(self, in_dims, num_classes):
        """Build the computation graph."""
        tf.reset_default_graph()
        tf.set_random_seed(1234)

        self.input = tf.placeholder(tf.float32, shape=(None, in_dims))
        logits = tf.layers.dense(self.input, num_classes)
        self.output = tf.argmax(logits, axis=1)

        self.labels = tf.placeholder(tf.int32, shape=None)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels, logits=logits)

        self.learning_rate = tf.placeholder(tf.float32)
        self.opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

        self.saver = tf.train.Saver()

    def train(self, train_data, valid_data, params,
              save_path="./tmp/model.ckpt"):
        """
        Train the neural network and save the model.

        If both validation input and labels are provided then the model's
        accuracy is evaluated on the validation data at the end of every epoch.

        Args:
            train_input: 2D numpy array of floats giving the training input.
                         Shape of array must be (data_points,
                         feature_vector_length)

            train_labels: 1D numpy array of ints giving the (enumerated)
                          labels.  Length must match the number of rows of
                          train_input.

            num_epochs (int): Number of epochs to train.

            learning_rate (float): Learning rate.

            batch_size (int): Batch size for training.

            valid_input: 2D numpy array of floats giving the validation
                         input. Shape of array must be (data_points,
                         feature_vector_length)

            valid_labels: 1D numpy array of ints giving the (enumerated)
                          labels.  Length must match the number of rows of
                          train_input.

            save_path: Filepath to save the model checkpoint to.

        Returns:
            Nothing.

        """
        np.random.seed(42)

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())

            for epoch in range(params['num_epochs']):

                print('Training epoch {}'.format(epoch))

                # Shuffle indices not data.
                shuffle_idx = np.arange(train_data['input'].shape[0])
                np.random.shuffle(shuffle_idx)

                for idx in range(0, len(shuffle_idx), params['batch_size']):

                    i = shuffle_idx[idx:idx+params['batch_size']]

                    fd = {self.input: train_data['input'][i, :],
                          self.labels: train_data['labels'][i],
                          self.learning_rate: params['learning_rate']}

                    _, loss = sess.run([self.opt, self.loss], feed_dict=fd)

                    print('Loss: {:.2f}'.format(loss[0]))

                # Validation test
                total_results = 0
                total_correct = 0

                for i in range(0, valid_data['input'].shape[0],
                               params['batch_size']):

                    fd = {self.input:
                          valid_data['input'][i:i+params['batch_size'], :]}

                    out = sess.run(self.output, feed_dict=fd)

                    correct = np.equal(out, valid_data['labels'][i:i+params['batch_size']])

                    total_results += correct.size
                    total_correct += np.sum(correct)

                    proportion_correct = total_correct/total_results

                    print('Validation accuracy: {:.2f}%'.format(
                        proportion_correct*100))

            self.saver.save(sess, save_path)

            print("Model saved in path: %s" % save_path)

    def predict(self, feature_vectors, restore_path="./tmp/model.ckpt"):
        """
        Predict the label of an input.

        Args:
            feature_vectors: 2D numpy array of feature vectors.  One row per
                             input.  Feature vector length must be the same
                             as the length used in the neural network's
                             training.
            restore_path: Path to model to restore.


        Returns: Integer corresponding to the prediction.

        """
        with tf.Session() as sess:

            self.saver.restore(sess, restore_path)
            print("Model restored from path: %s" % restore_path)

            fd = {self.input: feature_vectors}
            pred = sess.run(self.output, feed_dict=fd)

            return pred

    def evaluate(self, test_input, test_labels, batch_size=2,
                 restore_path="./tmp/model.ckpt"):
        """
        Evaluate the performance of the model on test data.

        Args:
            test_input: 2D numpy array of floats giving the training input.
                        Shape of array must be (data_points,
                        feature_vector_length)

            test_labels: 1D numpy array of ints giving the (enumerated)
                         labels.  Length must match the number of rows of
                         train_input.

            batch_size: Batch size for testing. Does not affect results,
                        only speed.

            restore_path: Filepath of checkpoint file from which to restore
                          the model.

        Returns:
            Nothing.

        """
        total_results = 0
        total_correct = 0

        with tf.Session() as sess:

            self.saver.restore(sess, restore_path)
            print("Model restored from path: %s" % restore_path)

            for i in range(0, test_input.shape[0], batch_size):

                fd = {self.input: test_input[i:i + batch_size, :]}
                out = sess.run(self.output, feed_dict=fd)

                correct = np.equal(out, test_labels[i:i+batch_size])

                total_results += correct.size
                total_correct += np.sum(correct)

            print('Test accuracy: {:.2f}%'.format(100 * total_correct /
                                                  total_results))
