import tensorflow as tf
import numpy as np
import cPickle
import os
from draw import DRAW

sess = tf.Session()

path = '/home/yao/DRAW/tmp/generate'

if os.path.exists(path):
  for fname in os.listdir(path):
    os.remove(path+'/'+fname)
else:
  os.mkdir(path)

with open('/home/yao/DRAW/mnist.pkl') as f:
  train_set, valid_set, test_set = cPickle.load(f)

data = train_set[0].reshape(-1, 28, 28)

hps={}
hps['iter'] = 40000
hps['lr'] = 1e-3
hps['N_r'] = 5
hps['N_w'] = 5
hps['nGlimpse'] = 48
hps['z_size'] = 100

draw = DRAW(sess, hps=hps)
draw.train(data)
draw.draw()

sess.close()