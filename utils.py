import numpy as np
import tensorflow as tf

encoder = tf.nn.rnn_cell.LSTMCell(1024)
decoder = tf.nn.rnn_cell.LSTMCell(1024)

def linear(name, x, nout, reuse):

  with tf.variable_scope(name, reuse=reuse):
    nin = x.get_shape().as_list()[1]
    w = tf.get_variable('W', [nin, nout])
    b = tf.get_variable('b', [nout])
    out = tf.matmul(x, w) + b

  return out

def err_image(x, canvas):

  sigmoid_convas = tf.sigmoid(canvas)
  err_image = x - sigmoid_convas
  
  return x - sigmoid_convas

def filterbank(N, A, B, gx, gy, delta, var):

  i = tf.range(1, N+1, dtype=tf.float32)
  mu_x = gx + (i- N/2.0 - 0.5)*delta
  mu_y = gy + (i- N/2.0 - 0.5)*delta

  whole_x = tf.range(1, A+1, dtype=tf.float32)
  whole_y = tf.range(1, B+1, dtype=tf.float32)

  a = whole_x - tf.expand_dims(mu_x, -1) 
  b = whole_y - tf.expand_dims(mu_y, -1)

  Fx = tf.exp(-tf.square(a)/tf.expand_dims(var, -1)/2.0)
  Fy = tf.exp(-tf.square(b)/tf.expand_dims(var, -1)/2.0)

  sum_Fx = tf.reduce_sum(Fx, axis=2, keep_dims=True)+1e-8
  sum_Fy = tf.reduce_sum(Fy, axis=2, keep_dims=True)+1e-8

  Fx = Fx/sum_Fx
  Fy = Fy/sum_Fy

  return Fx, Fy

def att_params(name, N, A, B, dec_hidden, reuse):

  params = linear(name, dec_hidden, 5, reuse)

  gx_ = params[:,0]
  gy_ = params[:,1]
  log_var = params[:,2]
  log_delta = params[:,3]
  log_gamma = params[:,4]

  var = tf.reshape(tf.exp(log_var), [-1, 1])
  delta = tf.reshape(tf.exp(log_delta), [-1, 1])
  gamma = tf.reshape(tf.exp(log_gamma), [-1, 1])

  gx = (A + 1)*(gx_ + 1) / 2
  gy = (B + 1)*(gy_ + 1) / 2
  gx = tf.reshape(gx, [-1, 1])
  gy = tf.reshape(gy, [-1, 1])
  delta = (tf.maximum(A, B)-1) * delta / (N-1)

  return gx, gy, var, delta, gamma

def _read(x, gamma, Fx, Fy):
  'utility function for read of generating or classifying task'

  N = Fx.get_shape().as_list()[1]
  patch = tf.batch_matmul(tf.batch_matmul(Fy, x),
              tf.transpose(Fx, [0, 2, 1]))

  return patch

def read(image, err_image, gamma, Fx, Fy, task='generating'):

  N = Fx.get_shape().as_list()[1]

  p_image = _read(image, gamma, Fx, Fy)
  p_image = tf.reshape(p_image, [-1, N*N]) # flatten

  if task == 'generating':
    p_err_image = _read(err_image, gamma, Fx, Fy)
    p_err_image = tf.reshape(p_err_image, [-1, N*N]) # flatten

  if task == 'generating':
    out = tf.mul(gamma, tf.concat(1, [p_image, p_err_image]))

  elif task == 'classifying':
    out = tf.mul(gamma, p_image)
  
  return out

def write(Fx, Fy, gamma, dec_hidden, reuse):

  N = Fx.get_shape().as_list()[1]

  write_patch = linear('w/write', dec_hidden, N*N, reuse)
  write_patch = tf.reshape(write_patch, [-1, N, N])

  write = tf.batch_matmul(tf.transpose(Fy, [0, 2, 1]), write_patch)
  write = tf.batch_matmul(write, Fx)
  write = tf.mul(1.0/tf.expand_dims(gamma, -1), write)

  return write

def encode(x, prev_state, reuse):
  with tf.variable_scope('r/encoder', reuse=reuse):
    enc_h, enc_state = encoder(x, prev_state)

  return enc_h, enc_state

def decode(x, prev_state, reuse):
  with tf.variable_scope('w/decoder', reuse=reuse):
    dec_h, dec_state = decoder(x, prev_state)

  return dec_h, dec_state

def latent_params(enc_h, z_size, reuse):

  mu = linear('w/latent_mu', enc_h, z_size, reuse)
  log_stddev = linear('w/latent_stddev', enc_h, z_size, reuse)
  stddev = tf.exp(log_stddev)

  batch_size = enc_h.get_shape().as_list()[0]
  normal = tf.random_normal([batch_size, z_size])

  sample = tf.add(mu, tf.mul(normal, stddev))

  square_mu = tf.square(mu)
  var = tf.square(stddev)
  log_var = 2*log_stddev

  return sample, square_mu, var, log_var

def binary_cross_entropy(p, q):

  eps = 1e-8

  return -tf.mul(p, tf.log(q+eps)) - tf.mul(1.0-p, tf.log(1.0-q+eps))