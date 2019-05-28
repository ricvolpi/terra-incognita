import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import lrelu
from tensorflow.contrib.slim.nets import resnet_v1, inception

class Model(object):
	"""Neural network
	"""
	def __init__(self, mode='train', arch='resnet_v1_50'):

		self.no_classes = 15
		self.img_size = 224
		self.no_channels = 3
		self.arch=arch	

	def encoder(self, images, reuse=False, is_training=True):

		if self.arch == 'resnet_v1_50':		
			with slim.arg_scope(resnet_v1.resnet_arg_scope()):
				net, end_points = resnet_v1.resnet_v1_50(inputs=images, num_classes=self.no_classes, reuse=reuse, is_training=is_training, global_pool=True)
		
		if self.arch == 'resnet_v1_101':		
			with slim.arg_scope(resnet_v1.resnet_arg_scope()):
				net, end_points = resnet_v1.resnet_v1_101(inputs=images, num_classes=self.no_classes, reuse=reuse, is_training=is_training, global_pool=True)

		return net


	def build_model(self):

		self.images = tf.placeholder(tf.float32, [None, self.img_size, self.img_size, self.no_channels], 'images') #images placeholder
		self.labels = tf.placeholder(tf.int64, [None], 'labels')
		self.is_training = tf.placeholder(tf.bool)

		#self.processed_images = inception_preprocessing.preprocess_for_train(self.images, self.img_size, self.img_size)		

		#logits that only depend on placeholder - no further perturbations
		self.logits = tf.squeeze(self.encoder(self.images, is_training=self.is_training))
		self.softmax_output = tf.nn.softmax(self.logits)
		
		#loss for the minimizer
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=tf.one_hot(self.labels,self.no_classes)))
		
		#for evaluation
		self.pred = tf.argmax(self.logits, 1)
		self.correct_pred = tf.equal(self.pred, self.labels)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

		self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0003) 

		t_vars = tf.trainable_variables()

		#minimizer
		self.train_op = slim.learning.create_train_op(self.loss, self.optimizer, variables_to_train = t_vars)
	
		loss_summary = tf.summary.scalar('loss', self.loss)
		accuracy_summary = tf.summary.scalar('accuracy', self.accuracy)
		self.summary_op = tf.summary.merge([loss_summary, accuracy_summary])		

