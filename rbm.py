from __future__ import print_function

import timeit

try:
	import PIL.Image as Image
except ImportError:
	import Image

import numpy as np
from datetime import datetime
import os

from utils import *

class RBM(object):

	def __init__(self,
	    n_visible=784,
	    n_hidden=500,
	    W=None,
	    hbias=None,
	    vbias=None,
	    numpy_rng=None
	):
		
		self.n_visible = n_visible
		self.n_hidden = n_hidden

		if numpy_rng is None:
			numpy_rng = np.random.RandomState(1234)

		if W is None:
			W = np.asarray(
				numpy_rng.uniform(
					 low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
		             high=4 * np.sqrt(6. / (n_hidden + n_visible)),
		            size=(n_visible, n_hidden)
		            )
				)

		if hbias is None: 
			hbias = np.zeros(n_hidden)

		if vbias is None:
			vbias = np.zeros(n_visible)
		
		self.numpy_rng = numpy_rng
		self.W = W
		self.hbias = hbias
		self.vbias = vbias

	def propup(self, vis):
		presig = np.dot(vis, self.W) + self.hbias
		return sigmoid(presig)

	def sample_h_given_v(self, v0_sample):
		h1_mean = self.propup(v0_sample)
		h1_sample = self.numpy_rng.binomial(size=h1_mean.shape,
		                                     n=1, p=h1_mean)
		return h1_mean, h1_sample

	def propdown(self, hid):
		presig = np.dot(hid, self.W.T) + self.vbias
		return sigmoid(presig)

	def sample_v_given_h(self, h0_sample):
		v1_mean = self.propdown(h0_sample)
		v1_sample = self.numpy_rng.binomial(size=v1_mean.shape,
		                                  n=1, p=v1_mean)
		return v1_mean, v1_sample

	def gibbs_hvh(self, h0_sample):
		v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
		h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
		return [v1_mean, v1_sample,
                h1_mean, h1_sample]

	def gibbs_vhv(self, v0_sample):
		h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
		v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
		return v1_sample

	# WHEN BATCH SIZE IS 1 NP.DOT COMPUTES SCALAR PRODUCT
	
	def train_mini_batch(self, batch, lr=0.1, persistent_chain=None, k=1):
		error, updates = self.get_error_updates(batch, lr, k)

		self.W += lr * updates[0]
		self.hbias += lr * updates[1]
		self.vbias += lr * updates[2]

		return error

	def get_error_updates(self, batch, lr=0.1, k=1):
		batch_size = batch.shape[0]
		# compute positive phase 
		ph_means, ph_samples = self.sample_h_given_v(batch)
		#negative phase with CD-k
		nv_means, nv_samples = self.sample_v_given_h(ph_samples)
		if k > 1:
			for i in range(k-1):
				nv_samples = self.gibbs_vhv(nv_samples)
		nh_means = self.propup(nv_samples)
		# parameter updates deltaW
		dw = (np.dot(batch.T, ph_means) - np.dot(nv_samples.T, nh_means))/batch_size
		dvbias = np.mean(batch - nv_samples, axis=0)
		dhbias = np.mean(ph_means - nh_means, axis=0)
		#batch error
		error = np.sum((batch - nv_means) ** 2) / batch_size

		return error, (dw, dhbias, dvbias)

	def train(self, learning_rate=0.1, epochs=15,
	         dataset='mnist.pkl.gz', training_size=500, batch_size=20, persistent=True,
	         pcd_length=1, n_chains=20, n_samples=10, 
	         output_folder='rbm_no_theano_plots'):
		print('start training')

		############# PREPARE DATA #############
		datasets = load_data(dataset)

		args = " lr {}, epochs {}, batch size {} ".format(learning_rate, epochs, batch_size)
		date = str(datetime.now().strftime('%d-%m %H-%M'))
		new_dir = output_folder+'/'+date+args
		if not os.path.isdir(new_dir):
			os.makedirs(new_dir)
		os.chdir(new_dir)
		# get training set 
		train_set_x, train_set_y = datasets[0]
		train_set_x = train_set_x[:training_size*100]
		test_set_x, test_set_y = datasets[2]
		print('data loaded!')
		n_train_batches = len(train_set_x) // batch_size
		print(n_train_batches, 'train batches for each epoch')

		if persistent:
			persistent_chain = np.zeros((batch_size, self.n_hidden))
		else: 
			persistent_chain = None

		########### START TRAINING #############
		plotting_time = 0.
		start_time = timeit.default_timer()

		# go through training epochs
		for epoch in range(epochs):
			print('going through epoch {}'.format(epoch))
			# self.numpy_rng.shuffle(train_set_x)
			
			error = 0.
			# go through training set with SGD
			for index in range(n_train_batches):
				batch = train_set_x[index * batch_size: (index + 1) * batch_size]
				error += self.train_mini_batch(batch, learning_rate, persistent_chain, pcd_length)
			
			print ('Epoch {}\'s error is {}'.format(epoch, error/n_train_batches))
			# print picture of weights
			plotting_start = timeit.default_timer()
			# print('plotting weights')
			image = Image.fromarray(
			    tile_raster_images(
			        X=self.W.T,
			        img_shape=(28, 28),
			        tile_shape=(14, 14),
			        tile_spacing=(1, 1)
			    )
			)
			image.save('filters_at_epoch_{}.png'.format(epoch))
			plotting_stop = timeit.default_timer()
			plotting_time += (plotting_stop - plotting_start)

		end_time = timeit.default_timer()
		pretraining_time = (end_time - start_time) - plotting_time

		print ('Training took %f minutes' % (pretraining_time / 60.))

		os.chdir('../')
		os.chdir('../')

		try:
			os.rmdir(new_dir)
		except OSError as ex:
			None
