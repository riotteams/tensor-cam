from base_cam import BaseCAM
import tensorflow as tf
from tensorflow.keras.applications import (ResNet50, VGG16, imagenet_utils)
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

class GradCAM(BaseCAM):
    
    def __init__(self, model, class_idx, layer_name=None):
        
        BaseCAM.__init__(self , model , layer_name = None)
        self.class_idx=class_idx
        
    
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError('Could not find 4D layer. Cannot apply GradCAM.')
        
    def compute_cam_features(self, image, eps=1e-8):

        grad_model = tf.keras.models.Model(inputs=[self.model.inputs],
                                           outputs=[self.model.get_layer(self.layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            conv_outs, predictions = grad_model(inputs)
            y_c = predictions[:, self.class_idx]

        grads = tape.gradient(y_c, conv_outs) 
        cast_conv_outputs = tf.cast(conv_outs > 0, tf.float32)
        cast_grads = tf.cast(grads > 0, tf.float32)
        guided_grads = cast_conv_outputs * cast_grads * grads 
        
        conv_outs = conv_outs[0]
        guided_grads = guided_grads[0]
        
        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        
        cam = tf.reduce_sum(tf.multiply(weights, conv_outs), axis=-1)
        
        w, h = image.shape[2], image.shape[1] 
        cam = resize(cam.numpy(), (w, h))
        cam /= np.max(cam)
        cam = (cam * 255).astype("uint8")
        return cam
