import numpy as np
import tensorflow as tf
import time
import pickle_data

start_time = time.time()

#Parameters
batch_size = 100
learning_rate = 0.005
max_steps = 1000

np.random.seed(1) #Comment to remove randomness

data = pickle_data.load_data()

#Initialize Tensorflow Graph
data_ph = tf.placeholder(tf.float32, shape=[None, 3072])
labels_ph = tf.placeholder(tf.int64, shape=[None])

#Tensor Variable
weights = tf.Variable(tf.zeros([3072, 10]))
biases = tf.Variable(tf.zeros([10]))

logits = tf.matmul(data_ph, weights) + biases

#Loss Function
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

#Training operation
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#Compare Prediction with true
correct_prediction = tf.equal(tf.argmax(logits, 1), labels_ph)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
Run Tensorflow Graph
'''

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	for i in range(max_steps):

		#Pick random batch from train data
		indices = np.random.choice(data['train_data'].shape[0], batch_size)

		train_batch = data['train_data'][indices]
		labels_batch = data['train_labels'][indices]

		#Print accuracy for evert 100 steps
		if i % 100 == 0:
			train_accu = sess.run(accuracy, feed_dict={data_ph : train_batch, labels_ph : labels_batch})
			print('Step: '+str(i) + ', Accuracy: '+str(train_accu))

		sess.run(train_step, feed_dict={data_ph : train_batch, labels_ph : labels_batch})

	test_accu = sess.run(accuracy, feed_dict={data_ph : data['test_data'], labels_ph : data['test_labels']})
	print('Accuracy: '+str(test_accu))

end_time = time.time()
print('Total time: {:5.2f}s'.format(end_time - start_time))