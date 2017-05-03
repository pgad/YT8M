''' author: samtenka
    changed: 2017-05-03
    created: 2017-02-17
    descr: Train baseline video tagger 
    usage: From `code/`, run the command
               python -m engine.train_tagger
           The training progress, specifically number of batches completed and
           validation loss, will update on screen. Cause a KeyboardInterrupt
           (press CTRL-c) to end training prematurely. Whether or not training
           ends early, the model will be saved to the checkpoint specified in
           `config.json`. 
'''
import numpy as np
import tensorflow as tf
from model.build_tagger import baseline 
from utils.config import get, is_file_prefix

def get_weights(saver, sess):
    ''' Load model weights if they were saved previously '''
    if is_file_prefix('TRAIN.CHECKPOINT'):
        saver.restore(sess, get('TRAIN.CHECKPOINT'))
        print('Yay! I restored weights from a saved model!')
    else:
        print('OK, I did not find a saved model, so I will start training from scratch!')

def report_training_progress(batch_index, orig, true, x_ent, Xs, Ys):
    ''' Update user on training progress '''
    if batch_index % 1: return
    print('starting batch number %d \033[100D\033[1A' % batch_index)
    if batch_index % 10: return
    error = x_ent.eval(feed_dict={orig: Xs, true:Ys})
    print('\n \t x_ent is about %f' % error)

def train_tagger(orig, true, x_ent, optimizer, Xs, Ys):
    ''' Train tagger '''
    try:
        for batch_index in range(get('TRAIN.NB_STEPS')):
            batch_i = np.random.randint(len(Xs), size=get('TRAIN.BATCH_SIZE')) 
            report_training_progress(batch_index, orig, true, x_ent, Xs, Ys)
            optimizer.run(feed_dict={orig: Xs[batch_i], true: Ys[batch_i]})
    except KeyboardInterrupt:
        print('OK, I will stop training even though I am not finished.') 
    
if __name__=='__main__':
    print('building model...')
    sess = tf.InteractiveSession() # start talking to tensorflow backend
    orig, pred = baseline() # fetch model layers
    true = tf.placeholder(tf.float32, [None, get('DATA.NB_CLASSES')])
    x_ent = - tf.multiply(tf.log(pred), true) - tf.multiply(tf.log(1-pred), 1-true) # loss func 
    optimizer = tf.train.AdamOptimizer(get('TRAIN.LEARNING_RATE')).minimize(x_ent) # define backward propagation
    sess.run(tf.global_variables_initializer()) # initialize globals
    saver = tf.train.Saver() # prepare to save model
    get_weights(saver, sess) # load model weights if they were saved previously
    
    print('loading data...')
    Xs = np.load(get('DATA.X')) 
    Ys = np.load(get('DATA.Y')) 
    
    print('training...')
    train_tagger(orig, true, x_ent, optimizer, Xs, Ys)
    
    print('saving trained model...\n')
    saver.save(sess, get('TRAIN.CHECKPOINT'))
    
