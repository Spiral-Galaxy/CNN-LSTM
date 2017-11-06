import tensorflow as tf
import numpy as np

'''
hyperparameters
'''

f_l = 7         #final size
f_w = 7

def MaxPool_Layer(value, kHeight, kWidth, strideX, strideY, name, padding='SAME'):
    with tf.name_scope(name):
        return tf.nn.max_pool(value = value,
                              ksize = [1,kHeight,kWidth,1],
                              strides = [1,strideX,strideY,1],
                              padding = padding,
                              name = name)

def DropOut_Layer(x, keep_prob, name = None):
    with tf.name_scope(name):
        return tf.nn.dropout(x = x,
                             keep_prob = keep_prob,
                             name = name)

def FC_Layer(x, input_size, output_size, ReluFlag, name):
    with tf.variable_scope(name) as scope:
        with tf.name_scope(name+'name'):
            w = tf.get_variable('w', shape = [input_size, output_size], dtype = 'float')
            b = tf.get_variable('b', shape = [output_size], dtype = 'float')
            out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
            if ReluFlag:
                return tf.nn.relu(out)
            else:
                return out

def Conv_Layer(x, kHeight, kWidth, strideX, strideY,
               featureNum, name, padding = 'SAME'):
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        with tf.name_scope(name+'name'):
            w = tf.get_variable('w', shape = [kHeight, kWidth, channel, featureNum])
            b = tf.get_variable('b', shape = [featureNum])
            featureMap = tf.nn.conv2d(x, w, strides = [1, strideX, strideY, 1], padding = padding)
            out = tf.nn.bias_add(featureMap, b)
            return tf.nn.relu(out, name = scope.name)



class VGG19(object):

    def __init__(self, x, keep_prob, classNum, skip, modelPath):
        self.X = x
        self.KEEPPROB = keep_prob
        self.CLASSNUM = classNum
        self.SKIP = skip
        self.MODELPATH = modelPath

    #    self.buildCNN(self.X)

    def buildCNN(self, X_in):
        conv1_1 = Conv_Layer(X_in, 3, 3, 1, 1, 64, 'conv1_1')
        conv1_2 = Conv_Layer(conv1_1, 3, 3, 1, 1, 64, 'conv1_2')
        pool1 = MaxPool_Layer(conv1_2, 2, 2, 2, 2, 'pool1')

        conv2_1 = Conv_Layer(pool1, 3, 3, 1, 1, 128, 'conv2_1')
        conv2_2 = Conv_Layer(conv2_1, 3, 3, 1, 1, 128, 'conv2_2')
        pool2 = MaxPool_Layer(conv2_2, 2, 2, 2, 2, 'pool2')

        conv3_1 = Conv_Layer(pool2, 3, 3, 1, 1, 256, 'conv3_1')
        conv3_2 = Conv_Layer(conv3_1, 3, 3, 1, 1, 256, 'conv3_2')
        conv3_3 = Conv_Layer(conv3_2, 3, 3, 1, 1, 256, 'conv3_3')
        conv3_4 = Conv_Layer(conv3_3, 3, 3, 1, 1, 256, 'conv3_4')
        pool3 = MaxPool_Layer(conv3_4, 2, 2, 2, 2, 'pool3')

        conv4_1 = Conv_Layer(pool3, 3, 3, 1, 1, 512, 'conv4_1')
        conv4_2 = Conv_Layer(conv4_1, 3, 3, 1, 1, 512, 'conv4_2')
        conv4_3 = Conv_Layer(conv4_2, 3, 3, 1, 1, 512, 'conv4_3')
        conv4_4 = Conv_Layer(conv4_3, 3, 3, 1, 1, 512, 'conv4_4')
        pool4 = MaxPool_Layer(conv4_4, 2, 2, 2, 2, 'pool4')

        conv5_1 = Conv_Layer(pool4, 3, 3, 1, 1, 512, 'conv5_1')
        conv5_2 = Conv_Layer(conv5_1, 3, 3, 1, 1, 512, 'conv5_2')
        conv5_3 = Conv_Layer(conv5_2, 3, 3, 1, 1, 512, 'conv5_3')
        conv5_4 = Conv_Layer(conv5_3, 3, 3, 1, 1, 512, 'conv5_4')
        pool5 = MaxPool_Layer(conv5_4, 2, 2, 2, 2, 'pool5')


        fcIn = tf.reshape(pool5, [-1, f_l * f_w * 512])
        fc6 = FC_Layer(fcIn, f_l * f_w * 512, 4096, True, 'fc6')
        dropout1 = DropOut_Layer(fc6, self.KEEPPROB, 'drop1')

        fc7 = FC_Layer(dropout1, 4096, 4096, True, 'fc7')
        dropout2 = DropOut_Layer(fc7, self.KEEPPROB, 'drop2')

        self.fc8 = FC_Layer(dropout2, 4096, self.CLASSNUM, True, 'fc8')

        return self.fc8
        # self.fc8.shape = (?, 512)

    def load_VGG(self, sess):
        wDict = np.load(self.MODELPATH, encoding = 'bytes').item()
        # load parameters for each layer
        for name in wDict:
            if name not in self.SKIP:
                with tf.variable_scope(name, reuse = True):
                    for p in wDict[name]:
                        if len(p.shape) == 1:
                            sess.run(tf.get_variable('b', trainable = True).assign(p))
                        else:
                            sess.run(tf.get_variable('w', trainable = True).assign(p))