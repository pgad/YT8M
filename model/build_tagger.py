''' author: samtenka
    changed: 2017-05-03
    created: 2017-02-12
    descr: build TensorFlow computation graph for multilabel video classification 
    usage: import this module via:
               from model.build_tagger import baseline 
'''
import numpy as np
import tensorflow as tf
from utils.config import get

def weight_variable(shape, sigma=0.1):
    ''' Return a TF variable initialized with independent gaussians of mean 0
        and supplied standard deviation
    '''
    W = tf.Variable(tf.random_normal(shape, stddev=sigma))
    return W

def bias_variable(shape, init=0.00):
    ''' Return a TF variable initialized to the supplied constant value '''
    b = tf.Variable(tf.constant(init, shape=shape))
    return b

def dense(x, in_dim, out_dim, activation=True): 
    ''' Dense layer with ELU activation '''
    W = weight_variable([in_dim, out_dim], sigma=1.0/in_dim**0.5)
    b = bias_variable([out_dim], init=1.0/in_dim)
    out = tf.matmul(x, W) + b 
    if activation:
        out = tf.nn.elu(out)
    return out

def baseline(nb_frames=get('DATA.NB_FRAMES'),
             nb_feats=get('DATA.NB_FEATS'),
             nb_classes=get('DATA.NB_CLASSES')):
    ''' Baseline architecture: feedforward fully connected network '''
    orig = tf.placeholder(tf.float32, shape=[None, nb_frames, nb_feats])

    h0 = tf.reduce_mean(orig, axis=1)
    h1 = dense(h0, nb_feats, 1024) 
    h2 = dense(h1, nb_feats, 256) 
    h3 = dense(h0, nb_feats, nb_classes, activation=False) 
    h3 = tf.nn.sigmoid(h3)  

    return orig, h3
