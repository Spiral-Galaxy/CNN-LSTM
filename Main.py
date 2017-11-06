import tensorflow as tf

from Vgg19 import *
from LSTM import *
from Config import *
from Get_Data import *

'''hyperparameter'''


def Build_Net(Videos):
    VGG_NET = VGG19(Videos, keep_prob, VGG_Class, skip, modelPath = Model_Path)

    return VGG_NET


def main():
    x = tf.placeholder(dtype = tf.float32, shape = [Seq_Len, None, H_size, L_size, C_size])
    y = tf.placeholder(dtype = tf.float32, shape = [None, Num_Class])

    # build vgg_net
    vgg_net = Build_Net(x[1,:,:,:])

    # reshape x so that can be used by lambda function
    #x_seq = get_seq_from_images(x)

    # connect vgg and LSTM and then get the output
    lstm_output = Joint_LSTM(x, vgg_net)

    # calculate the cost

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = lstm_output, labels = y))
    # train_op = tf.train.AdamOptimizer().minimize(cost)
    #
    # correct_label = tf.equal(tf.arg_max(lstm_output, 1), tf.arg_max(y, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_label, tf.float32))
    #
    # # global initializer
    #
    init = tf.global_variables_initializer()
    #
    # # load videos as data
    array = Get_Data(Video_Path)

    # run session
    with tf.Session() as sess:
        # initialize variables and load pre-trained data

        sess.run(init)
        vgg_net.load_VGG(sess)

        writer = tf.summary.FileWriter('./', sess.graph)
        writer.add_graph(sess.graph)
        step = 0
        i = 0

        print(sess.run(lstm_output, feed_dict = {
            x:array.reshape(151, 1, H_size, L_size, C_size)
        }))


main()
