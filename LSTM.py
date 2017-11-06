import tensorflow as tf
from Config import *

def build_LSTM():

    LSTM_Cell_1 = tf.contrib.rnn.LSTMCell(hid_units_1, forget_bias = 1.0, state_is_tuple = True)
    LSTM_Cell_2 = tf.contrib.rnn.LSTMCell(hid_units_2, forget_bias = 1.0, state_is_tuple = True)

    LSTM = tf.contrib.rnn.MultiRNNCell([LSTM_Cell_1, LSTM_Cell_2], state_is_tuple = True)

    return LSTM

def Joint_LSTM(sequence, vgg_net):
    inference_fn = lambda image : vgg_net.buildCNN(image)
    logit_seq = tf.map_fn(inference_fn, sequence, dtype = tf.float32, swap_memory = True)
    LSTM = build_LSTM()
    init_state = LSTM.zero_state(batch_size, dtype=tf.float32)

    s = [-1, 1, 1000]

    array = tf.concat((tf.reshape(logit_seq[0,:,:],s), tf.reshape(logit_seq[1,:,:],s)), 1)
    for i in range(2,logit_seq.shape[0]):
        array = tf.concat((array, tf.reshape(logit_seq[i,:,:],s)), 1)

    outputs, final_state = tf.nn.dynamic_rnn(LSTM, array, initial_state=init_state)

    with tf.name_scope('LSTM'):
        with tf.variable_scope('lstm') as scope:
            w = tf.Variable(tf.random_normal([hid_units_2, Num_Class]))
            b = tf.Variable(tf.constant(0.1, shape = [Num_Class]))

        outs = tf.matmul(final_state[-1][1], w) + b
        return outs