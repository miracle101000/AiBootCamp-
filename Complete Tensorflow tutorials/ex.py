import tensorflow as tf

# Create a Tensorflow constant tensor
tensor = tf.constant("Hello, Tensorflow!")

# Start a Tensorflow session
with tf.compat.v1.Session() as sess:
    # Run the Tensorflow operation to evaluate the tensor
    result = sess.run(tensor)
    # Print the result
    print(result.decode())
    

a = tf.constant(2)
b = tf.constant(3)    

# Define a Tensorflow operation to add two constants
add_op = tf.add(a, b)

# Start a Tensorflow session
with tf.compat.v1.Session() as sess:
    # Run the Tensorflow operation to evaluate the result
    result = sess.run(add_op)
    print("Result:", result)


# Create a Tensorflow placeholder for a 1D tensor of type float32
x =  tf.compat.v1.placeholder(tf.float32, shape=(None,))  

# Define a Tensorflow operation to compute the square of the input tensor
square_op = tf.square(x)
 
# Start a Tensorflow session
with tf.compat.v2.Session() as sess:
    # Run the Tensorflow operation with a placeholder input
    result = sess.run(square_op, feed_dict={x: [1, 2, 3]}) 
    print("Result:", result)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Create a Sequential model
model = Sequential([
    Dense(units=64, activation='relu', input_shape=(784,)),
    Dense(units=10, activation='softmax')
])    

# Compile the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    