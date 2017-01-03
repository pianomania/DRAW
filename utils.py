import numpy as np
import tensorflow as tf

def linear(name, x, nout, reuse):

	with tf.variable_scope(name, reuse=reuse):
		nin = x.get_shape().as_list()[1]
		w = tf.get_variable('W', [nin, nout])
		b = tf.get_variable('b', [nout])
		out = tf.matmul(x, w) + b

	return out

def filterbank(N, A, B, gx, gy, delta, sigma):

	i = tf.range(1, N+1, dtype=tf.float32)
	mu_x = gx + (i- N/2.0 - 0.5)*delta
	mu_y = gy + (i- N/2.0 - 0.5)*delta

	whole_x = tf.range(1, A+1, dtype=tf.float32)
	whole_y = tf.range(1, B+1, dtype=tf.float32)

	a = whole_x - tf.expand_dims(mu_x, -1) 
	b = whole_y - tf.expand_dims(mu_y, -1)

	Fx = tf.exp(-tf.square(a)/tf.expand_dims(sigma, -1)/2.0)
	Fy = tf.exp(-tf.square(b)/tf.expand_dims(sigma, -1)/2.0)

	sum_Fx = tf.reduce_sum(Fx, axis=2, keep_dims=True)
	sum_Fy = tf.reduce_sum(Fy, axis=2, keep_dims=True)

	Fx = Fx/sum_Fx
	Fy = Fy/sum_Fy

	return Fx, Fy

def att_params(name, A, B, dec_hidden, reuse):

	with tf.variable_scope(name):
		params = linear('linear_transform', dec_hidden, 5, reuse)

		gx_ = params[:,0]
		gy_ = params[:,1]
		log_sigma = params[:,2]
		log_delta = params[:,3]
		log_gamma = params[:,4]

		gx = (A + 1)*(gx_ + 1) / 2
		gy = (B + 1)*(gy_ + 1) / 2
		gx = tf.reshape(gx, [-1, 1])
		gy = tf.reshape(gy, [-1, 1])
		sigma = tf.reshape(tf.exp(log_sigma), [-1, 1])
		delta = tf.reshape(tf.exp(log_delta), [-1, 1])
		gamma = tf.reshape(tf.exp(log_gamma), [-1, 1, 1])

	return gx, gy, sigma, delta, gamma

def read(image, err_image, gamma, Fx, Fy):

	p_image = tf.batch_matmul(tf.batch_matmul(Fy, image),
														tf.transpose(Fx, [0, 2, 1]))
	p_err_image = tf.batch_matmul(tf.batch_matmul(Fy, err_image), 
																tf.transpose(Fx, [0, 2, 1]))

	return tf.mul(gamma, tf.concat(2, [p_image, p_err_image]))

def write(N, Fx, Fy, gamma, dec_hidden, reuse):

	with tf.variable_scope('write'):
		write_patch = linear('linear_transform', dec_hidden, N*N, reuse)
		write_patch = tf.reshape(write_patch, [-1, N, N])

		write = tf.batch_matmul(tf.transpose(Fy, [0, 2, 1]), write_patch)
		write = tf.batch_matmul(write, Fx)
		write = tf.mul(1.0/gamma, write)

	return write

