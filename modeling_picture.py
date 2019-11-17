# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""

import tensorflow as tf

from tensorflow.contrib.slim import nets

import preprocessing

slim = tf.contrib.slim
    
        
class Model(object):
    """xxx definition."""
    
    def __init__(self, num_classes, is_training,
                 fixed_resize_side,
                 default_image_size):
        """Constructor.
        
        Args:
            is_training: A boolean indicating whether the training version of
                computation graph should be constructed.
            num_classes: Number of classes.
        """
        self._num_classes = num_classes
        self._is_training = is_training
        self._fixed_resize_side = fixed_resize_side
        self._default_image_size = default_image_size
        
    @property
    def num_classes(self):
        return self._num_classes
        
    def preprocess(self, inputs):
        """preprocessing.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        preprocessed_inputs = preprocessing.preprocess_images(
            inputs, self._default_image_size, self._default_image_size, 
            resize_side_min=self._fixed_resize_side,
            is_training=self._is_training,
            border_expand=True, normalize=False,
            preserving_aspect_ratio_resize=False)
        preprocessed_inputs = tf.cast(preprocessed_inputs, tf.float32)
        return preprocessed_inputs
    
    def predict(self, preprocessed_inputs):
        """Predict prediction tensors from inputs tensor.
        
        Outputs of this function can be passed to loss or postprocess functions.
        
        Args:
            preprocessed_inputs: A float32 tensor with shape [batch_size,
                height, width, num_channels] representing a batch of images.
            
        Returns:
            prediction_dict: A dictionary holding prediction tensors to be
                passed to the Loss or Postprocess functions.
        """
        with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
            net, endpoints = nets.resnet_v1.resnet_v1_50(
                preprocessed_inputs, num_classes=None,
                is_training=self._is_training,reuse=tf.AUTO_REUSE)
        net = tf.squeeze(net, axis=[1, 2])
        logits = slim.fully_connected(net, num_outputs=self.num_classes,
                                      activation_fn=None, scope='Predict',reuse=tf.AUTO_REUSE)
        prediction_dict = {'logits': logits}
        return prediction_dict
    
    def postprocess(self, prediction_dict):
        """Convert predicted output tensors to final forms.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            **params: Additional keyword arguments for specific implementations
                of specified models.
                
        Returns:
            A dictionary containing the postprocessed results.
        """
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        predict_result = tf.argmax(logits, axis=1)
        # postprocessed_dict = {'logits': logits,
        #                       'classes': classes}
        return logits,predict_result
    
    def loss(self, prediction_dict, groundtruth_lists):
        """Compute scalar loss tensors with respect to provided groundtruth.
        
        Args:
            prediction_dict: A dictionary holding prediction tensors.
            groundtruth_lists_dict: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            A dictionary mapping strings (loss names) to scalar tensors
                representing loss values.sparse_softmax_cross_entropy
        """
        logits = prediction_dict['logits']
        slim.losses.sigmoid_cross_entropy(
            logits=logits, 
            multi_class_labels=groundtruth_lists,
            scope='Loss')
        loss = slim.losses.get_total_loss()
        loss_dict = {'loss': loss}
        return loss_dict
        
    def accuracy(self, predict_result, groundtruth_lists):
        """Calculate accuracy.
        
        Args:
            postprocessed_dict: A dictionary containing the postprocessed 
                results
            groundtruth_lists: A dict of tensors holding groundtruth
                information, with one entry for each image in the batch.
                
        Returns:
            accuracy: The scalar accuracy.
        """
        classes = predict_result#postprocessed_dict['classes']
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(classes, groundtruth_lists), dtype=tf.float32))
        return accuracy

    def do_predict(self,inputs,labels):
        preprocessed_inputs = self.preprocess(inputs)
        prediction_dict = self.predict(preprocessed_inputs)
        loss_dict = self.loss(prediction_dict, labels)
        loss = loss_dict['loss']
        logits,predict_result = self.postprocess(prediction_dict)
        return loss,logits,predict_result

