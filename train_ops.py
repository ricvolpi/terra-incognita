import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import numpy.random as npr
from ConfigParser import *
import os
import cPickle
import scipy.io
import sys
import glob
import json
from numpy.linalg import norm
from scipy import misc
from PIL import Image
import vgg_preprocessing
import skimage.transform
import sklearn.preprocessing
import pandas as pd
import time

from transformation_ops import TransfOps
from search_ops import SearchOps
import utils

class TrainOps(object):

	def __init__(self, model, exp_dir, run, arch):
		
		self.run = run
		self.arch = arch
		self.model = model
		self.exp_dir = exp_dir
		self.log_dir = os.path.join(self.exp_dir,'logs')
		self.model_save_path = os.path.join(self.exp_dir,'models/'+str(self.run))
		
		self.data_dir = './CaltechCameraTraps/ECCV2018'

		self.resnet_v1_50_ckpt = os.path.join('./resnet_v1_50.ckpt')
		self.resnet_v1_101_ckpt = os.path.join('./resnet_v1_101.ckpt')

		self.VGG_MEAN = [103.939, 116.779, 123.68]
		self.RGB_MEAN = [self.VGG_MEAN[2], self.VGG_MEAN[1], self.VGG_MEAN[0]]

		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
	
		if not os.path.exists(self.model_save_path):
			os.makedirs(self.model_save_path)
		
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth=True

		self.label_encoder = sklearn.preprocessing.LabelEncoder()
		self.label_encoder.fit(np.array([1,3,5,6,7,8,9,10,11,16,21,33,34,51,99])) # classes seen during training

		self.extract_all_metadata()

		self.transf_ops = TransfOps()
		self.search_ops = SearchOps()


	def extract_all_metadata(self):
		
		def extract_metadata(path_to_json):

			with open(path_to_json) as json_file:
				data = json.load(json_file)

			#image_paths = np.array([os.path.join(self.data_dir,'small_images/eccv_18_all_images_sm',str(item['image_id'])+'.jpg') for item in data['annotations']])
			image_paths = np.array([os.path.join(self.data_dir,'standard_images/eccv_18_cropped',str(item['image_id'])+'.jpg') for item in data['annotations']])
			labels = np.array([int(item['category_id']) for item in data['annotations']])


			image_paths = image_paths[labels!=30] 
			labels = labels[labels!=30] # not present in the training set
			labels = self.label_encoder.transform(labels)

			# Not used but possibly useful metadata below

			#locations = np.array([int(item['location']) for item in data['images']])
			#category_names = np.array([str(item['name']) for item in data['categories']])
			#category_labels = np.array([int(item['id']) for item in data['categories']])

			return image_paths, np.squeeze(labels)

		self.all_image_paths, self.all_labels = extract_metadata(os.path.join(self.data_dir,'annotations/CaltechCameraTrapsECCV18.json'))
		self.cis_test_image_paths, self.cis_test_labels = extract_metadata(os.path.join(self.data_dir,'annotations/cis_test_annotations.json'))
		self.cis_val_image_paths, self.cis_val_labels = extract_metadata(os.path.join(self.data_dir,'annotations/cis_val_annotations.json'))
		self.trans_test_image_paths, self.trans_test_labels = extract_metadata(os.path.join(self.data_dir,'annotations/trans_test_annotations.json'))
		self.trans_val_image_paths, self.trans_val_labels = extract_metadata(os.path.join(self.data_dir,'annotations/trans_val_annotations.json'))
		self.train_image_paths, self.train_labels = extract_metadata(os.path.join(self.data_dir,'annotations/train_annotations.json'))

		
	def load_exp_config(self):
	
		config = ConfigParser()
		config.read(os.path.join(self.exp_dir,'exp_configuration'))
	
		self.model.no_classes = 15

		if self.arch == 'resnet_v1_50' or self.arch == 'resnet_v1_101':
			self.model.img_size = 224


		self.train_iters = config.getint('MAIN_SETTINGS', 'train_iters')
		self.batch_size = config.getint('MAIN_SETTINGS', 'batch_size')
		self.model.batch_size = self.batch_size
	
		self.learning_rate = config.getfloat('MAIN_SETTINGS', 'learning_rate')


	def load_images(self, img_paths):

		images = np.zeros((len(img_paths), self.model.img_size, self.model.img_size, 3))

		for n,img_path in enumerate(img_paths):

			img = Image.open(img_path)

			img = img.resize((self.model.img_size, self.model.img_size), Image.ANTIALIAS)
			img = np.array(img, dtype=float)
				
			img = np.expand_dims(img, axis=0)

			if len(img.shape) == 3:
				img = np.stack((img,img,img), axis=3)

			images[n] = img

		if self.arch == 'resnet_v1_50' or self.arch == 'resnet_v1_101':
			images[:,:,:,0] -= self.RGB_MEAN[0]
			images[:,:,:,1] -= self.RGB_MEAN[1]
			images[:,:,:,2] -= self.RGB_MEAN[2]

		return images


	def train(self): 
	
		print 'Loading training data...'
		rand_idx = range(len(self.train_labels))
		npr.shuffle(rand_idx)
		self.train_labels = self.train_labels[rand_idx]
		self.train_image_paths = self.train_image_paths[rand_idx]
		train_labels = self.train_labels
		train_images = self.load_images(self.train_image_paths)
		print 'Done!'
		
		print 'Building model'
		self.model.mode='train'
		self.model.build_model()
		print 'Built'
	        
		transformations = [['identity']]
		levels = [[None]]	

		with tf.Session(config=self.config) as sess:
			tf.global_variables_initializer().run()

			if self.arch == 'resnet_v1_50':
				print ('Loading pretrained resnet_v1_50')
				variables_to_restore = slim.get_model_variables(scope='resnet_v1_50')
				variables_to_restore = [vv for vv in variables_to_restore if 'logits' not in vv.name]	    
				restorer = tf.train.Saver(variables_to_restore)
				restorer.restore(sess, self.resnet_v1_50_ckpt)

			elif self.arch == 'resnet_v1_101':
				print ('Loading pretrained resnet_v1_101')
				variables_to_restore = slim.get_model_variables(scope='resnet_v1_101')
				variables_to_restore = [vv for vv in variables_to_restore if 'logits' not in vv.name]	    
				restorer = tf.train.Saver(variables_to_restore)
				restorer.restore(sess, self.resnet_v1_101_ckpt)

			saver = tf.train.Saver()
		
			summary_writer = tf.summary.FileWriter(logdir=self.log_dir, graph=tf.get_default_graph())

			print 'Training'
			for t in range(self.train_iters):
	
				i = t % int(len(train_images) / self.batch_size)

				#current batch of images and labels
				batch_images = np.copy(train_images[i*self.batch_size:(i+1)*self.batch_size])
				batch_labels = np.copy(train_labels[i*self.batch_size:(i+1)*self.batch_size])

				feed_dict = {self.model.images: batch_images, self.model.labels: batch_labels, self.model.is_training: True}

				#running a step of gradient descent
				sess.run(self.model.train_op, feed_dict) 

				if t % 50 == 0: #evaluating the model

					rand_idxs = np.random.permutation(train_images.shape[0])[:100]
			
					train_acc, train_loss = sess.run(fetches=[self.model.accuracy, self.model.loss], 
														feed_dict={self.model.images: train_images[rand_idxs], 
																self.model.labels: train_labels[rand_idxs],
																self.model.is_training: False})

					summary = sess.run(self.model.summary_op, feed_dict)
					summary_writer.add_summary(summary, t)		


					print ('Run: [%d] Step: [%d/%d] train_loss: [%.4f] train_acc: [%.4f]'%(int(self.run), t+1, self.train_iters, train_loss, train_acc))

				if i == 0 and t != 0: # shuffling dataset after each epoch
					rand_idx = range(len(train_labels))
					npr.shuffle(rand_idx)
					train_labels = train_labels[rand_idx]
					train_images = train_images[rand_idx]
	
				if (t+1) % 500 == 0: 
					print 'Saving'
					saver.save(sess, os.path.join(self.model_save_path, 'encoder'))


	def test_foo(self, images, labels, sess):

		N = 20 #set accordingly to GPU memory
		accuracy = 0
		loss = 0
		preds = []
		softmax_output = np.zeros((images.shape[0], self.model.no_classes))

		start = 0

		for images_batch, labels_batch in zip(np.array_split(images, N), np.array_split(labels, N)):

			feed_dict = {self.model.images: images_batch, self.model.labels: labels_batch, self.model.is_training: False} 
			accuracy_tmp, loss_tmp, pred_tmp, softmax_tmp = sess.run([self.model.accuracy, self.model.loss, self.model.pred, self.model.softmax_output], feed_dict) 

			accuracy += accuracy_tmp/float(N)
			loss += loss_tmp/float(N)
			softmax_output[start:start+images_batch.shape[0]] += softmax_tmp
			preds += pred_tmp.tolist()

			start += images_batch.shape[0]

		correct_preds = (np.array(preds)==labels).astype(int)

		entropy = np.sum(-np.log(softmax_output) * softmax_output, 1)
		entropy[~np.isfinite(entropy)] = 0.

		return accuracy, loss, correct_preds, entropy


	def test(self):

		test_labels = self.trans_test_labels
		test_image_paths = self.trans_test_image_paths

		rand_idx = range(len(test_labels))
		npr.shuffle(rand_idx)
		test_labels = test_labels[rand_idx]
		test_image_paths = test_image_paths[rand_idx]

		print 'Loading data'
		test_labels = test_labels[:1000]
		test_images = self.load_images(test_image_paths[:1000])
		print 'Loaded'
				
		print 'Building model'
		self.model.mode='train'
		self.model.build_model()
		print 'Built'

		with tf.Session() as sess:
		
			tf.global_variables_initializer().run()
		
			print ('Loading pre-trained model.')
			variables_to_restore = slim.get_model_variables()
			restorer = tf.train.Saver(variables_to_restore)
			restorer.restore(sess, os.path.join(self.model_save_path,'encoder'))
				
			print 'Calculating accuracy'
		
			target_accuracy, target_loss, _, _ = self.test_foo(test_images, test_labels, sess)
		
			print ('Target accuracy: [%.4f], Target loss: [%.4f]'%(target_accuracy, target_loss))
	   

if __name__=='__main__':

	print 'To be implemented.'


