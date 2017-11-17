import tensorflow as tf
import time
from datetime import datetime
import pickle_data
import two_layer_fc

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

data = pickle_data.load_data()

#Initialize Tensorflow Graph
data_ph = tf.placeholder(tf.float32, shape=[None, 3072], name='data')
labels_ph = tf.placeholder(tf.int64, shape=[None], name='labels')

logits = two_layer_fc.inference(data_ph, 3072, FLAGS.hidden1, 10, reg_constant=FLAGS.reg_constant)

global_step = tf.Variable(0, name="global_step", trainable=False)

accuracy = two_layer_fc.evaluation(logits, labels_ph)
saver = tf.train.Saver()

with tf.Session() as sess:
	ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)

	if ckpt and ckpt.model_checkpoint_path:
		print('Restoring checkpoint')
		saver.restore(sess, ckpt.model_checkpoint_path)
		current_step = tf.train.global_step(sess, global_step)
		print('Current step: {}'.format(current_step))

	test_accu = sess.run(accuracy, feed_dict={data_ph : data['test_data'], labels_ph : data['test_labels']})
	print('Accuracy: '+str(test_accu))

end_time = time.time()
print('Total time: {:5.2f}s'.format(end_time - start_time))