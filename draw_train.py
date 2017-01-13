from utils import filterbank, att_params, read, write, err_image, latent_params, binary_cross_entropy, encode, decode, encoder, decoder
import tensorflow as tf
import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sess = tf.InteractiveSession()

if os.path.exists('/home/yao/DRAW/tmp/generate'):
  for fname in os.listdir('/home/yao/DRAW/tmp/generate'):
    os.remove('/home/yao/DRAW/tmp/generate'+'/'+fname)
else:
  os.mkdir('/home/yao/DRAW/tmp/generate')

with open('/home/yao/DRAW/mnist.pkl') as f:
  train_set, valid_set, test_set = cPickle.load(f)

data = train_set[0].reshape(-1, 28, 28)

batch_size = 64
iteration = 30000
GS = tf.Variable(0, trainable=False, name='global_step')
lr = tf.train.exponential_decay(1e-3, GS, 5000, 0.75)

N_r = 2
N_w = 5
A = 28.0
B = 28.0
maxT = 32
z_size = 100

enc_state = encoder.zero_state(batch_size, dtype=tf.float32)
dec_state = decoder.zero_state(batch_size, dtype=tf.float32)
dec_h = dec_state[1]

canvas = tf.zeros((64,28,28))
image = tf.placeholder(tf.float32, [batch_size, 28, 28])

canvas_series = []
gx_series = []
gy_series = []
delta_series = []
var_series = []

for step in xrange(maxT):
  reuse = not (step == 0)

  image_hat = err_image(image, tf.sigmoid(canvas)) 

  gx_r, gy_r, var_r, delta_r, gamma_r = att_params('r/params', N_r, A, B, enc_state[1], reuse)
  Fx_r, Fy_r = filterbank(N_r, A, B, gx_r, gy_r, delta_r, var_r)
  r = read(image, image_hat, gamma_r, Fx_r, Fy_r)
  encode_in = tf.concat(1, [r, dec_h])
  enc_h, enc_state = encode(encode_in, enc_state, reuse)

  z, z_mu, z_var, z_log_var = latent_params(enc_h, z_size, reuse)

  dec_h, dec_state = decode(z, dec_state, reuse)

  gx_w, gy_w, var_w, delta_w, gamma_w = att_params('w/params', N_w, A, B, dec_state[1], reuse)
  Fx_w, Fy_w= filterbank(N_w, A, B, gx_w, gy_w, delta_w, var_w)
  w = write(Fx_w, Fy_w, gamma_w, dec_h, reuse)
  canvas += w

  canvas_series.append(canvas) # testing
  gx_series.append(gx_w)
  gy_series.append(gy_w)
  delta_series.append(delta_w)
  var_series.append(var_w)

  tf.add_to_collection('z_mu', z_mu)
  tf.add_to_collection('z_var', z_var)
  tf.add_to_collection('z_log_var', z_log_var)
  tf.summary.histogram('z_distribution', z) # 

z_mu_sum = tf.reduce_sum(tf.get_collection('z_mu'), axis=0) 
z_var_sum = tf.reduce_sum(tf.get_collection('z_var'), axis=0)
z_log_var_sum = tf.reduce_sum(tf.get_collection('z_log_var'), axis=0)

batch_loss_z = 0.5*tf.reduce_sum(z_mu_sum + z_var_sum - z_log_var_sum - maxT, axis=1) 

loss_z = tf.reduce_mean(batch_loss_z)

x_entropy = binary_cross_entropy(image, tf.sigmoid(canvas))
loss_x = tf.reduce_mean(tf.reduce_sum(x_entropy, axis=[1,2]))

loss = loss_x + loss_z

optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5, beta2=0.75)
gvs = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_norm(grad, 5.0), var) for grad, var in gvs]
train_op = optimizer.apply_gradients(capped_gvs, global_step=GS)


tf.summary.scalar('loss', loss)
tf.summary.scalar('loss_x', loss_x)
tf.summary.scalar('loss_z', loss_z)
tf.summary.scalar('z_mu', tf.reduce_mean(z_mu_sum))
tf.summary.scalar('z_var', tf.reduce_mean(z_var_sum))
tf.summary.scalar('z_log_var', tf.reduce_mean(z_log_var_sum))
summary_op = tf.merge_all_summaries()
summary_writer = tf.summary.FileWriter('/home/yao/DRAW/tmp/generate', sess.graph)

sess.run(tf.global_variables_initializer())

for i in range(iteration):

  idx = np.random.permutation(data.shape[0])[:64]

  loss_eval ,_, _, _, _, _, summary, _ = sess.run([loss, loss_x, loss_z, z_mu_sum, 
                                           z_var_sum, z_log_var_sum, summary_op, train_op], 
      feed_dict={image: data[idx]})
  
  summary_writer.add_summary(summary, i)
  print loss_eval

# generating image
for step in range(maxT):
  c = sess.run(canvas_series[step], feed_dict={image: data[idx]})
  c = tf.sigmoid(c).eval()

  gx, gy, delta, var = sess.run([gx_series[step], gy_series[step], delta_series[step], var_series[step]], feed_dict={image: data[idx]})

  k=0
  fig, ax =plt.subplots(8,8)
  for i in range(8):
    for j in range(8):
      ax[i][j].imshow(c[k], cmap='gray')
      xx = gx[k] + (0.5 - N_w/2.0)*delta[k]
      yy = gy[k] + (0.5 - N_w/2.0)*delta[k]
      ww = delta[k] * (N_w-1)
      rect = patches.Rectangle((xx, yy), ww, ww,linewidth=np.sqrt(var[k]), edgecolor='r', facecolor='none')
      ax[i][j].add_patch(rect)
      ax[i][j].set_axis_off()
      k += 1
  plt.tight_layout(h_pad=-1.5,w_pad=-14)
  plt.savefig('/home/yao/DRAW/tmp/generate/test %i.png' % step)
  plt.close()
