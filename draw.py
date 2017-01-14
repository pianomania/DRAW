from utils import *
import tensorflow as tf
import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class DRAW(object):

  def __init__(self, sess, N_r=12, N_w=12, A=28.0, B=28.0, maxT=32, z_size=100):

    self.sess = sess
    self.N_r = N_r
    self.N_w = N_w
    self.A = A
    self.B = B
    self.maxT = maxT
    self.z_size = z_size

    self.iteration = 1000
    self.batch_size = 64
    self.GS = tf.Variable(0, trainable=False, name='global_stap')
    self.lr = 1e-3

    enc_state = encoder.zero_state(self.batch_size, dtype=tf.float32)
    dec_state = decoder.zero_state(self.batch_size, dtype=tf.float32)
    enc_h = enc_state[1]
    dec_h = dec_state[1]

    canvas = tf.zeros((64,28,28))

    self.image = tf.placeholder(tf.float32, [self.batch_size, 28, 28])

    for step in xrange(maxT):
      reuse = not (step == 0)

      image_hat = err_image(self.image, tf.sigmoid(canvas)) 

      gx_r, gy_r, var_r, delta_r, gamma_r = att_params('r/params', N_r, A, B, enc_h, reuse)
      Fx_r, Fy_r = filterbank(N_r, A, B, gx_r, gy_r, delta_r, var_r)
      r = read(self.image, image_hat, gamma_r, Fx_r, Fy_r)
      encode_in = tf.concat(1, [r, dec_h])
      enc_h, enc_state = encode(encode_in, enc_state, reuse)

      z, z_mu, z_var, z_log_var = latent_params(enc_h, z_size, reuse)

      dec_h, dec_state = decode(z, dec_state, reuse)

      gx_w, gy_w, var_w, delta_w, gamma_w = att_params('w/params', N_w, A, B, dec_h, reuse)
      Fx_w, Fy_w= filterbank(N_w, A, B, gx_w, gy_w, delta_w, var_w)
      w = write(Fx_w, Fy_w, gamma_w, dec_h, reuse)
      canvas += w

      tf.add_to_collection('z_mu', z_mu)
      tf.add_to_collection('z_var', z_var)
      tf.add_to_collection('z_log_var', z_log_var)
      tf.summary.histogram('z_distribution', z) # 

    z_mu_sum = tf.reduce_sum(tf.get_collection('z_mu'), axis=0) 
    z_var_sum = tf.reduce_sum(tf.get_collection('z_var'), axis=0)
    z_log_var_sum = tf.reduce_sum(tf.get_collection('z_log_var'), axis=0)

    batch_loss_z = 0.5*tf.reduce_sum(z_mu_sum + z_var_sum - z_log_var_sum - maxT, axis=1) 

    self.loss_z = tf.reduce_mean(batch_loss_z)

    x_entropy = binary_cross_entropy(self.image, tf.sigmoid(canvas))
    self.loss_x = tf.reduce_mean(tf.reduce_sum(x_entropy, axis=[1,2]))

    self.loss = self.loss_x + self.loss_z

    optimizer = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5, beta2=0.75)
    gvs = optimizer.compute_gradients(self.loss)
    capped_gvs = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in gvs]
    self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.GS)


    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('loss_x', self.loss_x)
    tf.summary.scalar('loss_z', self.loss_z)
    self.summary_op = tf.merge_all_summaries()
    self.summary_writer = tf.summary.FileWriter('/home/yao/DRAW/tmp/generate', sess.graph)

    self.generate_()

  def generate_(self):

    self.z_plh = tf.placeholder(tf.float32, [self.batch_size, self.z_size])
    self.c_plh = tf.placeholder(tf.float32, [self.batch_size, 256])
    self.h_plh = tf.placeholder(tf.float32, [self.batch_size, 256])

    N_w = self.N_w
    A = self.A
    B = self.B

    dec_h, dec_state = decode(self.z_plh, (self.c_plh, self.h_plh), True)

    gx_w, gy_w, var_w, delta_w, gamma_w = \
        att_params('w/params', N_w, A, B, dec_h, True)
    
    Fx_w, Fy_w= filterbank(N_w, A, B, gx_w, gy_w, delta_w, var_w)
    
    self.w = write(Fx_w, Fy_w, gamma_w, dec_h, True)
    self.c_state = dec_state[0]
    self.h_state = dec_state[1]
    self.gx_w = gx_w
    self.gy_w = gy_w
    self.var_w = var_w
    self.delta_w = delta_w

  def train(self, data):
    self.sess.run(tf.global_variables_initializer())

    for i in range(self.iteration):

      idx = np.random.permutation(data.shape[0])[:64]

      loss_eval, loss_x, loss_z, summary, _ = self.sess.run(
          [self.loss, self.loss_x, self.loss_z, self.summary_op, self.train_op], 
          feed_dict={self.image: data[idx]})
      
      self.summary_writer.add_summary(summary, i)
      print loss_eval

  def draw(self):

    canvas = np.zeros((self.batch_size, 28, 28))
    c_state = np.zeros((self.batch_size, 256))
    h_state = np.zeros((self.batch_size, 256))
    np.seterr(over='ignore')

    for step in range(self.maxT):
      
      w, gx, gy, delta, var, c_state, h_state = self.sess.run(
        [self.w,
        self.gx_w,
        self.gy_w,
        self.delta_w,
        self.var_w,
        self.c_state,
        self.h_state],
        feed_dict={self.z_plh: np.random.randn(self.batch_size, self.z_size),
                   self.c_plh: c_state,
                   self.h_plh: h_state})

      canvas += w
      c = 1/(1+np.exp(-canvas))

      k=0
      fig, ax =plt.subplots(8,8)
      for i in range(8):
        for j in range(8):
          ax[i][j].imshow(c[k], cmap='gray')
          xx = gx[k] + (0.5 - self.N_w/2.0)*delta[k]
          yy = gy[k] + (0.5 - self.N_w/2.0)*delta[k]
          ww = delta[k] * (self.N_w-1)
          rect = patches.Rectangle((xx, yy), ww, ww,linewidth=np.sqrt(var[k]), edgecolor='r', facecolor='none')
          ax[i][j].add_patch(rect)
          ax[i][j].set_axis_off()
          k += 1
      plt.tight_layout(h_pad=-1.5,w_pad=-14)
      plt.savefig('/home/yao/DRAW/tmp/generate/test%i.png' % step)
      plt.close()
