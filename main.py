import tensorflow as tf
from model import Model
from train_ops import TrainOps
import glob
import os
import cPickle

import numpy.random as npr
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('gpu', '0', "GPU to used")
flags.DEFINE_string('run', '0', "run")
flags.DEFINE_string('exp_dir', 'exp_dir', "Experiment directory")
flags.DEFINE_string('seed', '213', "Experiment directory")
flags.DEFINE_string('mode', 'mode', "Experiment directory")
flags.DEFINE_string('arch', 'resnet_v1_50', "Architecture to be used")

FLAGS = flags.FLAGS

def main(_):

	GPU_ID = FLAGS.gpu
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152 on stackoverflow
	os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

	RUN = FLAGS.run
	EXP_DIR = FLAGS.exp_dir
	SEED = int(FLAGS.seed)
	ARCH = FLAGS.arch

	model = Model(arch=ARCH)
	train_ops = TrainOps(model, EXP_DIR, RUN, ARCH)
	train_ops.load_exp_config()

	if FLAGS.mode=='train':
		print 'Training'
		train_ops.train(seed=SEED)       

	if FLAGS.mode=='test':
		print 'Testing'
		train_ops.test() 

if __name__ == '__main__':
	tf.app.run()










