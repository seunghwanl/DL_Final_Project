
# coding: utf-8

# In[ ]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
from math import sqrt

import numpy as np
import tensorflow as tf


# In[ ]:

N_DIGITS = 10  # Number of classes.


# In[73]:

def resnet_model_fn(features, labels, mode): 
    
    with tf.device('/device:GPU:0'):
        Group = namedtuple('Group', ['density', 'num_filters'])
    
        groups = [
            Group(6, 32), Group(12, 32),
            Group(24, 32), Group(16, 32)
        ]
    
        #groups = [
        #    Group(6, 64), Group(4, 64),
        #    Group(4, 64), Group(4, 64)
        #]
    
        input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])
        input_layer = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), input_layer)

        with tf.variable_scope('conv_layer1'):
            bn = tf.layers.batch_normalization(inputs=input_layer, training=mode==tf.estimator.ModeKeys.TRAIN)
            bn_relu = tf.nn.relu(bn)
            conv = tf.layers.conv2d(
                bn_relu,
                filters=16,
                kernel_size=3,
                padding='same')
            net = tf.layers.dropout(inputs=conv, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
            tf.summary.histogram('weights_conv_layer_1', net)
        #[100,32,32,16]

        with tf.variable_scope('conv_layer2'):
            net = tf.layers.conv2d(
                net,
                filters=groups[0].num_filters,
                kernel_size=1,
                padding='valid')
            tf.summary.histogram('weights_conv_layer_2', net) 


        for group_i, group in enumerate(groups):
            input_net = net
            for layer_i in range(group.density):
                name = 'group_%d/layer_%d' % (group_i, layer_i)
                with tf.variable_scope(name + '/dense'):
                    bn = tf.layers.batch_normalization(inputs=net, training=mode==tf.estimator.ModeKeys.TRAIN)
                    bn_relu = tf.nn.relu(bn)
                    conv = tf.layers.conv2d(
                        bn_relu,
                        filters=group.num_filters,
                        kernel_size=3,
                        padding='same')
                    conv_dropout = tf.layers.dropout(inputs=conv, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                    #net = tf.concat(axis=3, values=(net, conv_dropout)) 
                    tf.summary.histogram('weights_conv_group_%d/_layer_%d_before' % (group_i, layer_i), conv_dropout)
                    net = net + conv_dropout
                    tf.summary.histogram('weights_conv_group_%d/_layer_%d_after' % (group_i, layer_i), net)
            with tf.variable_scope('group_%d/conv_reduce' % group_i):    
                bn = tf.layers.batch_normalization(inputs=net, training=mode==tf.estimator.ModeKeys.TRAIN)
                bn_relu = tf.nn.relu(bn)
                input_dim = int(input_net.get_shape()[-1])
                conv = tf.layers.conv2d(
                    bn_relu,
                    filters=input_dim,
                    kernel_size=1,
                    padding='valid')
                conv_dropout = tf.layers.dropout(inputs=conv, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
                tf.summary.histogram('weights_conv_group_%d/_conv_reduce_before' % group_i, conv_dropout)
            net = conv_dropout + input_net
            tf.summary.histogram('weights_conv_group_%d/_conv_reduce_after' % group_i, net)
            try:
                next_group = groups[group_i + 1]
                with tf.variable_scope('group_%d/conv_upscale' % group_i):
                    net = tf.layers.conv2d(
                        net,
                        filters=next_group.num_filters,
                        kernel_size=1,
                        padding='same',
                        activation=None,
                        bias_initializer=None)
                tf.summary.histogram('weights_conv_group_%d/_conv_upscale' % group_i, net) 
            except IndexError:
                pass
        
        net_shape = net.get_shape().as_list()
        net = tf.nn.avg_pool(
            net,
            ksize=[1, net_shape[1], net_shape[2], 1],
            strides=[1, 1, 1, 1],
            padding='VALID')

        net_shape = net.get_shape().as_list()
        net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
        tf.summary.histogram('weights_avg_pool_flattened', net) 
        
        logits = tf.layers.dense(net, N_DIGITS, activation=None)

        predicted_classes = tf.argmax(logits, 1)
        predictions = {
                'classes': predicted_classes,
                'probabilities': tf.nn.softmax(logits)
        }
        
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), N_DIGITS, 1, 0)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:  
            learning_rate_with_decay = tf.train.exponential_decay(learning_rate=0.1, global_step=tf.train.get_global_step(), decay_steps=10000, decay_rate=0.8, staircase=True)
            optimizer = tf.train.AdagradOptimizer(learning_rate_with_decay)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch norm
            with tf.control_dependencies(extra_update_ops):
                train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            correct_predictions = tf.equal(predictions["classes"], tf.argmax(onehot_labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            tf.summary.scalar("Training_Accuracy", accuracy)
            
            total_parameters = 0
            for variable in tf.global_variables():
                shape = variable.get_shape()
                variable_parametes = 1
                for dim in shape:
                    variable_parametes *= dim.value
                total_parameters += variable_parametes
            tf.summary.scalar("total_params", total_parameters)
                
            tensors_log = {
                "loss": loss,
                "train_accuracy": accuracy
            }
            
            logging_hook = tf.train.LoggingTensorHook(tensors=tensors_log, every_n_iter=1000)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[logging_hook])
  
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)
        }
       
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


# In[ ]:

(X_train, y_train), (X_test, y_test) = tf.contrib.keras.datasets.cifar10.load_data()

X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32).squeeze()
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32).squeeze()


# In[74]:

classifier = tf.estimator.Estimator(model_fn=resnet_model_fn, model_dir="/Users/snehanagaraj/Documents/DL/Project/WDN/wide_dense_net_v6/cifar")

tf.logging.set_verbosity(tf.logging.INFO)  # Show training logs.

# Train model and save summaries into logdir.
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": X_train},
    y=y_train,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
classifier.train(input_fn=train_input_fn, steps=1)

# Calculate accuracy.
#test_input_fn = tf.estimator.inputs.numpy_input_fn(
#    x={"x": X_test},
#    y=y_test,
#    num_epochs=1,
#    shuffle=False)
#scores = classifier.evaluate(input_fn=test_input_fn)
#print('Accuracy: {0:f}'.format(scores['accuracy']))

