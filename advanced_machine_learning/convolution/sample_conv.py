import tensorflow as tf
from tensorflow.python.keras.layers import MaxPooling2D

# Define the input tensor with float32 data type
X = tf.constant([
    [0, 0, 2, 2],
    [0, 0, 2, 2],
    [0, 0, 2, 2],
    [0, 0, 2, 2]
], dtype=tf.float32)

# Reshape the tensor to have batch and channel dimensions
X_reshaped = tf.reshape(X, [1, 4, 4, 1])

# Define the custom kernel
kernel = tf.constant([
    [[-1], [-1], [1]],  
    [[-1], [-1], [1]],  
    [[-1], [-1], [1]] 
], dtype=tf.float32)

# Reshape the kernel to the format expected by tf.nn.conv2d [filter_height, filter_width, in_channels, out_channels]
kernel = tf.reshape(kernel, [3, 3, 1, 1])

# Apply convolution with padding='SAME' (which adds padding of 1) and stride of 2
conv_result = tf.nn.conv2d(
    input=X_reshaped,
    filters=kernel,
    strides=[1, 1, 1, 1],  # Stride of 2 in both height and width dimensions
    padding='VALID'         # Adds padding as needed (effectively padding of 1)
)

# Reshape back to 2D
result = tf.reshape(conv_result, [-1, conv_result.shape[2]])

print("Original array:")
print(X.numpy())
print("\nConvolution kernel:")
print(tf.reshape(kernel, [3, 3, 1]).numpy())
print("\nAfter convolution (kernel size 3x3, stride 1x1, padding=1):")
print(result.numpy())