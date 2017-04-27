import tensorflow as tf
import numpy as np

from globals import num_classes, batch_size
from utils import weight_var, bias_var


class Layer:
    def __init__(self, m1, m2, m_id):
        self.m_id = m_id
        self.w = weight_var([m1, m2])
        self.b = bias_var([m2])

    def forward(self, x):
        return tf.nn.relu(tf.matmul(x, self.w) + self.b)


class ANN:
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = []
        self.w = None
        self.b = None
        self.tf_x = None
        self.tf_y = None
        self.y = None

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer.forward(z)

        return tf.matmul(z, self.w) + self.b

    def build(self, data):
        # Get shape and initialise network
        k = num_classes
        n, d = data.images_train.shape

        m1 = d
        id_count = 0
        for m2 in self.hidden_layer_sizes:
            layer = Layer(m1, m2, id_count)
            self.layers.append(layer)
            m1 = m2
            id_count += 1

        # Get params for last layer - will be useful for forward prop
        # Because m1 is now m2, we can use it instead
        self.w = weight_var([m1, k])
        self.b = bias_var([k])

        self.tf_x = tf.placeholder(tf.float32, shape=[None, d])
        self.tf_y = tf.placeholder(tf.float32, shape=[None, k])
        self.y = self.forward(self.tf_x)

    def fit(self, data, learning_rate=10e-4, epochs=200):
        sess = tf.InteractiveSession()

        # Define operations
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_y, logits=self.y))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.tf_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_accuracy = 0.0

        for i in xrange(epochs):
            for x_batch, y_batch in data.iterate_batches():
                train_step.run(feed_dict={self.tf_x: x_batch, self.tf_y: y_batch})

            if i % 10:
                train_accuracy = accuracy.eval(feed_dict={self.tf_x: data.images_val, self.tf_y: data.labels_val})
                print 'Step:', i, 'Validation accuracy:', train_accuracy
                if train_accuracy > best_accuracy:
                    best_accuracy = train_accuracy
                    model_path = saver.save(sess, 'models/model', global_step=10)
                    print 'Best score! Model saved in', model_path

    def predict(self, x):
        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        saver.restore(sess, 'models/model-10')

        predicted_labels = np.zeros(x.shape[0])
        predict_action = tf.argmax(self.forward(self.tf_x), 1)
        for i in xrange(x.shape[0] // batch_size):
            predicted_labels[i*batch_size: (i+1) * batch_size] = \
                predict_action.eval(feed_dict={self.tf_x: x[i * batch_size: (i + 1) * batch_size]})

        return predicted_labels
