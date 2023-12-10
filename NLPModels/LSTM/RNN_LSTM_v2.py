# coding: utf-8

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

GPUconfig = tf.compat.v1.ConfigProto();
GPUconfig.gpu_options.per_process_gpu_memory_fraction = 0.4

import DataMaker
import numpy as np
import time
import sys
import pickle
import argparse
import copy
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import math_ops
import collections

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

# With this, you disable the default activate eager execution and you don't need to touch the code much more.
tf.compat.v1.disable_eager_execution();

_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))

class LSTMStateTuple(_LSTMStateTuple):
    """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
    Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state
    and `h` is the output.
    Only used when `state_is_tuple=True`.
    """
    __slots__ = ()

    @property
    def dtype(self):
        (c, h) = self
        if c.dtype != h.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                (str(c.dtype), str(h.dtype)))
        return c.dtype


class LSTM_RNNCell(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, init_scale, hidden_size, reuse=True):
        #self._num_units = num_units
        #self._activation = activation or math_ops.tanh
        self._hidden_size = hidden_size
        self.init_scale = init_scale;

    @property
    def state_size(self):
        return LSTMStateTuple(self._hidden_size, self._hidden_size)

    @property
    def output_size(self):
        return self._hidden_size

    def __call__(self, inputs, state, scope=None):
        """LSTM Zaremba network:
        where Wx = """
        c, h = state
        #print(state.shape)
        with tf.compat.v1.variable_scope(scope or "LSTM"):

            with tf.compat.v1.variable_scope("Gates"):
                W_i = tf.compat.v1.get_variable('W_i', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                U_i = tf.compat.v1.get_variable('U_i', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                b_i = tf.compat.v1.get_variable('b_i', [self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                i = tf.sigmoid(tf.matmul(inputs, W_i) + tf.matmul(h, U_i) + b_i)
                W_f = tf.compat.v1.get_variable('W_f', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                U_f = tf.compat.v1.get_variable('U_f', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                b_f = tf.compat.v1.get_variable('b_f', [self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                f = tf.sigmoid(tf.matmul(inputs, W_f) + tf.matmul(h, U_f) + b_f)
                W_o = tf.compat.v1.get_variable('W_o', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                U_o = tf.compat.v1.get_variable('U_o', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                b_o = tf.compat.v1.get_variable('b_o', [self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                #o = tf.sigmoid(tf.matmul(state, U_o) + b_o)
                o = tf.sigmoid(tf.matmul(inputs, W_o) + tf.matmul(h, U_o) + b_o)
            with tf.compat.v1.variable_scope("Candidate"):
                W_g = tf.compat.v1.get_variable('W_g', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                U_g = tf.compat.v1.get_variable('U_g', [self._hidden_size, self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                b_g = tf.compat.v1.get_variable('b_g', [self._hidden_size], initializer=tf.compat.v1.random_uniform_initializer(-self.init_scale, self.init_scale))
                #g = f*c + i*tf.tanh(tf.matmul(inputs, W_g) + tf.matmul(state, U_g) + b_g)
                g = tf.tanh(tf.matmul(inputs, W_g) + tf.matmul(h, U_g) + b_g)

            new_c = (c * f + i * g)
            new_h = tf.tanh(new_c) * o

            new_state = LSTMStateTuple(new_c, new_h)
            #output = tf.softmax(W_y*mt + b_y)
        return new_h, new_state

#_LSTMStateTuple = collections.namedtuple("LSTMStateTuple", ("c", "h"))


class batch_producer(object):
    '''Slice the raw data into batches'''
    def __init__(self, raw_data, batch_size, num_steps):
        self.raw_data = raw_data
        self.batch_size = batch_size
        self.num_steps = num_steps

        self.batch_len = len(self.raw_data) // self.batch_size
        self.data = np.reshape(self.raw_data[0 : self.batch_size * self.batch_len],
                            (self.batch_size, self.batch_len))

        self.epoch_size = (self.batch_len - 1) // self.num_steps
        self.i = 0

    def __next__(self):
        if self.i < self.epoch_size:
            # batch_x and batch_y are of shape [batch_size, num_steps]
            batch_x = self.data[::,
                self.i * self.num_steps : (self.i + 1) * self.num_steps : ]
            batch_y = self.data[::,
                self.i * self.num_steps + 1 : (self.i + 1) * self.num_steps + 1 : ]
            self.i += 1
            return (batch_x, batch_y)
        else:
            raise StopIteration()

    def __iter__(self):
        return self

class MyModel:
    def __init__(self, config, is_train):
        # get hyperparameters
        self.batch_size = batch_size = config.BatchSize
        self.num_steps = num_steps = config.StepsPerEpoch
        init_scale = config.init_scale
        word_emb_dim = hidden_size = config.hidden_size
        word_vocab_size = config.VocabularySize
        #initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        # language model
        #with tf.variable_scope('model', initializer=initializer):
        # embedding matrix
        with tf.compat.v1.variable_scope('word_embedding_scope', reuse=tf.compat.v1.AUTO_REUSE):
            word_embedding = tf.compat.v1.get_variable("word_embedding", [word_vocab_size, word_emb_dim])

        # placeholders for training data and labels
        self.x = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
        self.y = tf.compat.v1.placeholder(tf.int32, [batch_size, num_steps])
        # we first embed words ...
        words_embedded = tf.nn.embedding_lookup(word_embedding, self.x)
        if is_train:
            rnn_input = tf.nn.dropout(words_embedded, rate=1 - (config.keep_prob))#dropOut = words_embedd
            # ... and then process it with a stack of two LSTMs
        if is_train:
            rnn_input = tf.unstack(rnn_input, axis=1)
        else:
            rnn_input = tf.unstack(words_embedded, axis=1)
        # basic RNN cell
        cell1 = LSTM_RNNCell(config.init_scale,hidden_size)
        if is_train:
            cell1 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=config.keep_prob)
        cell2 = LSTM_RNNCell(config.init_scale,hidden_size)
        if is_train:
            cell2 = tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell2, output_keep_prob=config.keep_prob)
        cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([cell1, cell2])
        
        self.init_state = cell.zero_state(batch_size, dtype=tf.float32)

        state = self.init_state
        outputs, self.state = tf.compat.v1.nn.static_rnn(
            cell,
            rnn_input,
            dtype=tf.float32,
            initial_state=self.init_state)
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, hidden_size])
        # softmax layer
        weights = tf.compat.v1.get_variable('weights', [hidden_size, word_vocab_size], dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [word_vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, weights) + biases
        ###########################################################################
        # loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        #      [logits],
        #      [tf.reshape(self.y, [-1])],
        #      [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.reshape(self.y, [-1]),
            logits,
            from_logits=True
        )
        loss = tf.reduce_mean(loss)
        ###########################################################################
        self.cost = cost = tf.reduce_sum(loss) / batch_size

        # training
        self.lr = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        tvars = tf.compat.v1.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(self.lr)
        ###########################################################################
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars),
        #                                           global_step = tf.contrib.framework.get_or_create_global_step())
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        ###########################################################################
        self.new_lr = tf.compat.v1.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self.lr_update = tf.compat.v1.assign(self.lr, self.new_lr)

        self.final_state =state

    def assign_lr(self, session, lr_value):
        session.run(self.lr_update, feed_dict={self.new_lr: lr_value})

#session.run(lr_update, feed_dict={new_lr: lr_value})


# In[ ]:

###########################################################################
# def model_size():
#   '''finds the total number of trainable variables a.k.a. model size'''
#   params = tf.compat.v1.trainable_variables()
#   size = 0
#   for x in params:
#     sz = 1
#     for dim in x.get_shape():
#       sz *= dim.value
#     size += sz
#   return size
def model_size():
    '''finds the total number of trainable variables a.k.a. model size'''
    params = tf.compat.v1.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.shape:
            sz *= dim
        size += sz
    return size
###########################################################################

def run_epoch(sess, m_train, raw_data, train_op, config, is_train=False, lr=None):

    start_time = time.time()
    if is_train: 
        m_train.assign_lr(sess, lr)

    iters = 0
    costs = 0
    state_val = sess.run(m_train.init_state)

    batches = batch_producer(raw_data, config.BatchSize, config.StepsPerEpoch)
    
    for batch_idx, (batch_x, batch_y) in enumerate(batches):
        # run the model on current batch
        _, c, state_val = sess.run(
            [train_op, m_train.cost, m_train.state],
            feed_dict={m_train.x: batch_x, m_train.y: batch_y,
                       m_train.init_state: state_val})
        
        # Check if cost is NaN or Inf
        if np.isnan(c) or np.isinf(c):
            print(f"[DEBUG]: NaN or Inf value detected in cost at batch {batch_idx}. Exiting...");
            return float('inf');

        costs += c
        iters += config.StepsPerEpoch
        
        step = iters // config.StepsPerEpoch
        if is_train and step % (batches.epoch_size // 10) == 10:
            elapsed_time = (time.time() - start_time)
            print(f'\t[..]: Progress: {step * 1.0 / batches.epoch_size:.2f}, ',
                  f'Train Perplexity: {np.exp(costs / iters):.3f}, ',
                  f'Elapsed Time: {elapsed_time:.3f}s, ',
                  f'Speed: {iters * config.BatchSize / elapsed_time:.2f} wps');
            
    if iters == 0:
        print("Warning: No iterations were run. Returning inf perplexity.")
        return float('inf')
    else:
        return np.exp(costs / iters)
    
def TrainAndValidLSTMModel(ModelConfig, TrainData, ValidData):
    PatienceCounter = 0;
    #print("[DEBUG]: Define a uniform random initializer with the specified initialization scale from the config")
    initializer = tf.compat.v1.random_uniform_initializer(-ModelConfig.init_scale, ModelConfig.init_scale)
    #print("[DEBUG]: Define the training model within a TensorFlow variable scope")
    with tf.compat.v1.variable_scope('Model', reuse=False, initializer=initializer):
        m_train = MyModel(config=ModelConfig, is_train=True)
    #print('[DEBUG]: Model size is: ', model_size())
    #print("[DEBUG]: Define the validation model with the same variable scope as the training model")
    with tf.compat.v1.variable_scope('Model', reuse=True, initializer=initializer):
        m_valid = MyModel(config=ModelConfig, is_train=False)
    #print("[DEBUG]: Create a saver object to save and restore TensorFlow model variables")
    saver = tf.compat.v1.train.Saver()

    #print("[DEBUG]: Initialize global variables")
    init = tf.compat.v1.global_variables_initializer()
    #print("[DEBUG]: Get the learning rate from the configuration")
    learning_rate = ModelConfig.learning_rate
    #print("[DEBUG]: Start a TensorFlow session with GPU configuration")
    with tf.compat.v1.Session(config=GPUconfig) as sess:
        #print("[DEBUG]: Run the variables initializer")
        sess.run(init)
        #print("[DEBUG]: Set initial values for previous and best validation set perplexity")
        prev_valid_ppl = float('inf')
        best_valid_ppl = float('inf')
        #print("[DEBUG]: Train the model for a number of epochs..")
        for Epoch in range(ModelConfig.MaxEpochNumber):
            #print("[DEBUG]: Run one training epoch and get the perplexity")
            train_ppl = run_epoch(
                sess, m_train, TrainData, m_train.train_op, ModelConfig, is_train=True,lr=learning_rate)
            # Print the epoch number, training set perplexity, and current learning rate
            print('[DEBUG]: epoch: %3d' % (Epoch + 1), end=': ')
            print('train ppl = %.3f' % train_ppl, end=', ')
            print('lr = %.3f' % learning_rate, end=', ')
            # Get the validation set perplexity
            valid_ppl = run_epoch(sess, m_valid, ValidData, tf.no_op(), ModelConfig, is_train=False)
            
            Delta = best_valid_ppl - ModelConfig.MinDelta;
            if valid_ppl < Delta:
                best_valid_ppl = valid_ppl
                PatienceCounter = 0  # Reset patience
                # Save model if validation performance improves
                save_path = saver.save(sess, ModelConfig.ModelCheckPoint)
                print(f"Valid ppl improved     | valid_ppl={valid_ppl}")
            else:
                PatienceCounter += 1  # No improvement in validation performance
                print(f"Valid ppl NOT improved | valid_ppl={valid_ppl}")
                learning_rate *= ModelConfig.LearningRateDecay
            
            if PatienceCounter >= ModelConfig.EpochPatience:
                print(f"[DEBUG]: Early stop. PatienceCounter={ModelConfig.EpochPatience} reached");
                break;
            if Epoch >= ModelConfig.MaxEpochNumber:
                print(f"[DEBUG]: Early stop. MaxEpochNumber={ModelConfig.MaxEpochNumber} reached");
                break;
    tf.compat.v1.reset_default_graph()



def TestLSTMModel(ModelConfig, TestData, DataLabel):
    initializer = tf.compat.v1.random_uniform_initializer(-ModelConfig.init_scale, ModelConfig.init_scale)
    #print("[DEBUG]: Define the test model with the same variable scope as the training model")
    with tf.compat.v1.variable_scope('Model', reuse=tf.compat.v1.AUTO_REUSE, initializer=initializer):
        m_test = MyModel(config=ModelConfig, is_train=False)
    saver = tf.train.Saver()
    with tf.compat.v1.Session(config=GPUconfig) as sess:
        # Restore variables from disk.
        saver.restore(sess, ModelConfig.ModelCheckPoint)
        # Get test set perplexity
        test_ppl = run_epoch(sess, m_test, TestData, tf.no_op(), ModelConfig, is_train=False)
        print('[DEBUG][%s]: Test set perplexity = %.3f' % (DataLabel,test_ppl))

        