import tensorflow as tf

print("Num GPUs Available: ", len( tf.config.experimental.list_physical_devices('GPU') )) # For TensorFlow/Keras

print( tf.config.experimental.list_physical_devices('GPU') )