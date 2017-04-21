import tensorflow as tf

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

    def forward(self, x):
        z = x
        for layer in self.layers:
            z = layer.forward(z)

        return tf.matmul(z, self.w) + self.b

    def predict(self, x):
        return tf.argmax(self.forward(x), 1)

    def fit(self, data, learning_rate=10e-4, epochs=200):
        # Get shape and initialise network
        k = data.labels_train.shape[1]
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

        tf_x = tf.placeholder(tf.float32, shape=[None, d])
        tf_y = tf.placeholder(tf.float32, shape=[None, k])
        y = self.forward(tf_x)

        # Define operations
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_y, logits=y))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            for i in xrange(epochs):
                for x_batch, y_batch in data.iterate_batches():
                    train_step.run(feed_dict={tf_x: x_batch, tf_y: y_batch})

                if i % 10:
                    train_accuracy = accuracy.eval(feed_dict={tf_x: data.images_val, tf_y: data.labels_val})
                    print 'Step:', i, 'Validation accuracy:', train_accuracy
