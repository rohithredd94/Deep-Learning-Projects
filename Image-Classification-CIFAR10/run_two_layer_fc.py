import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import pickle_data
import two_layer_fc

def gen_batch(data, batch_size, num_iter):
  data = np.array(data)
  index = len(data)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data)):
      index = 0
      shuffled_indices = np.random.permutation(np.arange(len(data)))
      data = data[shuffled_indices]
    yield data[index:index + batch_size]

#Tensorflow Model Parameters for neural net
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for the training.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('batch_size', 400,
  'Batch size. Must divide dataset sizes without remainder.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr, value))
print()

start_time = time.time()

#Directory to put tensorflow logs
logdir = FLAGS.train_dir + '/' + datetime.now().strftime('%Y%m%d-%H%M%S') + '/'

np.random.seed(1) #Comment to remove randomness

#Load CIFAR10 data
data = pickle_data.load_data()

#Initialize Tensorflow Graph
data_ph = tf.placeholder(tf.float32, shape=[None, 3072], name='data')
labels_ph = tf.placeholder(tf.int64, shape=[None], name='labels')

#Classifiers Result
logits = two_layer_fc.inference(data_ph, 3072, FLAGS.hidden1, 10, reg_constant=FLAGS.reg_constant)

#Loss Function
loss = two_layer_fc.loss(logits, labels_ph)

#Training Operation
train_step = two_layer_fc.training(loss, FLAGS.learning_rate)

accuracy = two_layer_fc.evaluation(logits, labels_ph)

#Summary data for tensorboard
summary = tf.summary.merge_all()
saver = tf.train.Saver()

'''
Run Tensorflow Graph
'''

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	summary_writer = tf.summary.FileWriter(logdir, sess.graph)

	#Generate input data batches
	data_zip = zip(data['train_data'], data['train_labels'])
	batches = gen_batch(list(data_zip), FLAGS.batch_size, FLAGS.max_steps)

	for i in range(FLAGS.max_steps):
		batch = next(batches)
		batch_data, batch_labels = zip(*batch)
		feed_dict = {
			data_ph: batch_data,
			labels_ph: batch_labels
		}

		#Print accuracy for every 100 steps
		if i % 100 == 0:
			train_accu = sess.run(accuracy, feed_dict=feed_dict)
			print('Step: '+str(i) + ', Accuracy: '+str(train_accu))
			smry_str = sess.run(summary, feed_dict=feed_dict)
			summary_writer.add_summary(smry_str, i)

		sess.run([train_step, loss], feed_dict=feed_dict)

		#Save Checkpoint
		if (i+1) % 1000 == 0:
			ckpt_file = os.path.join(FLAGS.train_dir, 'checkpoint')
			saver.save(sess, ckpt_file, global_step=i)
			print('Checkpoint Saved')

	test_accu = sess.run(accuracy, feed_dict={data_ph : data['test_data'], labels_ph : data['test_labels']})
	print('Accuracy: '+str(test_accu))


end_time = time.time()
print('Total time: {:5.2f}s'.format(end_time - start_time))