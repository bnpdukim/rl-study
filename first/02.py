import tensorflow as tf

n_inputs = 4
n_hidden = 4
n_outputs = 1
initializer = tf.variance_scaling_initializer()

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
outpus = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat