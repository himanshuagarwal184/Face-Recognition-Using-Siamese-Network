# CUSTOM L1 DISTANCE LAYER
# Done to load the trained model
#dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer


#l1 distance layer 
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self,input_embedding,validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)