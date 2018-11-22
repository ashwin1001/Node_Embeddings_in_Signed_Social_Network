from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

# Import the data
data = open("../status_values_similarity/Similarity2.txt","r")
X = []
for line in data:
        data = [float(i) for i in line.strip().split()]
        X.append(data)
        
X = np.array(X)
print(X)

input = X
input_data = input


n_samp, num_input = X.shape 
print(X.shape)
print(n_samp)

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = min(50, n_samp)

display_step = 100
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

def calculateLoss(y_true,y_pred):
	if y_true == 1:
		return (y_true - y_pred)*10
	elif y_true == 0:
		return (y_true - y_pred)/10
	else:
		return (y_true - y_pred)

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
sample = 3

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

batch_size = min(50, n_samp)
n_rounds = 5000

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    file2=open('loss2.txt','w')
    # Training
    for i in range(n_rounds):
        sample = np.random.randint(n_samp, size=batch_size)
        batch_xs = input_data[sample][:]
        _, l = sess.run([optimizer, loss], feed_dict={X: batch_xs})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))
            file2.write(str(i)+'\t'+str(l)+'\n')
    file1=open("embedding2.txt","w")
    for i in range(num_input):
        file1.write(str(sess.run(encoder_op, feed_dict={X: input_data})[i])+'\n')
    file1.close()
    file2.close()
