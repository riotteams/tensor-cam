import tensorflow as tf
from tensorflow.keras.applications import (ResNet50, VGG16, imagenet_utils)
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


class BaseCAM:
    def __init__(self, model, layer_name=None):
        self.model = model
        self.layer_name = layer_name
        if not layer_name:
            self.layer_name = self.find_target_layer()

    def get_image(image_path):
        orig = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(orig)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)
        return image, orig

    def compute_cam_features(self, image):
        cam_model = tf.keras.models.Model(inputs=[self.model.inputs],
                                          outputs=[self.model.get_layer(self.layer_name).output, self.model.output])
        features, outputs = cam_model.predict(image)
        global_avg_pool_weights = self.model.layers[-1].get_weights()[0]
        image_feature = features[0]
        cam_features = resize(image_feature, (224, 224))
        pred = np.argmax(outputs[0])
        cam_weights = global_avg_pool_weights[:, pred]
        cam_output = np.dot(cam_features, cam_weights)
        return cam_output

    def plot_images(image, cam):
        plt.figure(figsize=(7, 7))
        plt.axis('off')
        plt.imshow(image)
        plt.show()
        plt.figure(figsize=(7, 7))
        plt.axis('off')
        plt.imshow(cam, cmap='jet')
        plt.show()
        plt.figure(figsize=(7, 7))
        plt.axis('off')
        plt.imshow(image)
        plt.imshow(cam, cmap='jet', alpha=0.5)
        plt.show()
