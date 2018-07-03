import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()

in_units = 784
h1_units = 300
h2_units = 100
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.truncated_normal([h1_units, h2_units], stddev=0.1))
b2 = tf.Variable(tf.zeros([h2_units]))
W3 = tf.Variable(tf.zeros([h2_units, 10]))
b3 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

hidden2 = tf.nn.relu(tf.matmul(hidden1_drop, W2) + b2)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

y = tf.nn.softmax(tf.matmul(hidden2_drop, W3) + b3)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean((-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])))
train_step = tf.train.AdagradOptimizer(0.05).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(32)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
    train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1})
    train_loss=cross_entropy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1})
    print("step %d, train accuracy %g ,%g" % (i, train_accuracy,train_loss))

print(accuracy.eval({x: mnist.train.images, y_: mnist.train.labels, keep_prob: 1.0}))
