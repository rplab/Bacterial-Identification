

from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from time import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from skimage.transform import resize


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv_3d(x, w):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def conv_layer(x, kernel=[2, 5, 5], num_in=1, num_out=16):
    w_conv1 = weight_variable([kernel[0], kernel[1], kernel[2], num_in, num_out])
    b_conv1 = bias_variable([num_out])
    return tf.nn.leaky_relu(conv_3d(x, w_conv1) + b_conv1)


def max_pool(x):
    global pool_count
    pool_count += 1
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


def dense_layer(x, num_in=2 * 7 * 7 * 32, num_out=1024):
    w_fc1 = weight_variable([num_in, num_out])
    b_fc1 = bias_variable([num_out])
    h_pool2_flat = tf.reshape(x, [-1, num_in])
    dense = tf.nn.leaky_relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    return dense


def softmax_layer(x, num_in=1024, num_classes=2):
    w_fc2 = weight_variable([num_in, num_classes])
    b_fc2 = bias_variable([num_classes])
    return tf.nn.softmax(tf.matmul(x, w_fc2) + b_fc2)


def rotate_data(data_in, labels):
    data = []
    lab = []
    for el in range(len(data_in)):
        image = data_in[el]
        lable = labels[el]
        # if len(image[0]) == 30 and len(image[1]) == 30:
        if np.random.randint(0, 2) == 0:
            image = np.fliplr(image)
        if np.random.randint(0, 2) == 0:
            image = np.flipud(image)
        if np.random.randint(0, 2) == 0:
            image = np.array(image)[:, :, ::-1]
        if np.random.randint(0, 2) == 0:
            image = np.transpose(image, (0, 2, 1))
        data.append(image)
        lab.append(lable)
    return data, lab


#  SAVE DATA AS COMPRESSED NPZ AND MAKE SURE ALL CUBES ARE RESIZED TO 8X28x28 USE: from skimage.transform import resize
def extract_data(file_name):
    loaded = np.load(file_name)
    data, labels = loaded['data'], loaded['labels']
    train_data_out, test_data_out, train_labels_out, test_labels_out = train_test_split(data, labels)
    return train_data_out, test_data_out, train_labels_out, test_labels_out


#
#                               LOAD DATA, CREATE TRAIN AND TEST SET
#

file_loc = '/media/parthasarathy/Bast/pseudomonas_data_labels.npz'
print('Importing and splitting data: ')
train_data, test_data, train_labels, test_labels = extract_data(file_loc)

#
#                               HYPERPARAMETERS
#

L1 = 32  # number of convolutions for first layer
L2 = 64  # number of convolutions for second layer
L3 = 1024  # number of neurons for dense layer
l_rate = .0001  # learning rate
dropout_rate = 0.5  # rate of neurons dropped off dense layer during training
epochs = 120  # number of times we loop through training data
batch_size = 120  # number of images per batch
rotate = True

#
#                               CREATE THE TENSORFLOW GRAPH
#

num_labels = 2
cube_length = 28*28*8
session_tf = tf.InteractiveSession()
pool_count = 0
flat_cube = tf.placeholder(tf.float32, shape=[None, cube_length])
y_ = tf.placeholder(tf.float32, shape=[None, num_labels])
cube = tf.reshape(flat_cube, [-1, 8, 28, 28, 1])  # [batch size, depth, height, width, channels]
#   first layer
conv_l1 = conv_layer(cube, num_in=1, num_out=L1)  # num_in is input size, num_out is output size.
#   pooling
pooling_l1 = max_pool(conv_l1)
#   second layer
conv_l2 = conv_layer(pooling_l1, num_in=L1, num_out=L2)
#   pooling
pooling_l2 = max_pool(conv_l2)
#   dense layer - neurons fully connecting all conv neuron outputs
dense_neurons = L3
dense_l3 = dense_layer(pooling_l2, num_in=int(cube_length / (2 * pool_count) ** 3) * L2, num_out=dense_neurons)
keep_prob = tf.placeholder(tf.float32)
dropped_l3 = tf.nn.dropout(dense_l3, keep_prob)
#   softmax
outputNeurons = softmax_layer(dropped_l3, num_in=dense_neurons, num_classes=num_labels)  # soft max to predict
#   loss - optimizer - evaluation
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(outputNeurons + 1e-10), reduction_indices=[1]))
optimizer = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(outputNeurons, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session_tf.run(tf.global_variables_initializer())

#
#                               TRAIN THE NETWORK
#

train_accuracy_list = []
train_time0 = time()
print('Training neural network:')
for epoch in range(epochs):
    print('epoch: ' + str(epoch))
    temp_data, temp_labels = rotate_data(train_data, train_labels)  # Randomly flip and rotate images for each epoch
    temp_data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in temp_data]
    for batch in range(len(train_data) // batch_size):
        offset = (batch * batch_size) % len(train_data)
        batch_data = temp_data[offset:(offset + batch_size)]
        batch_labels = temp_labels[offset:(offset + batch_size)]
        optimizer.run(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: dropout_rate})
        if batch == 0:  # Output train accuracy at the beginning of each epoch
            train_accuracy = accuracy.eval(feed_dict={
                flat_cube: batch_data, y_: batch_labels, keep_prob: 1.0})
            cross_ent = cross_entropy.eval(feed_dict={
                flat_cube: batch_data, y_: batch_labels, keep_prob: 1.0})
            print("training accuracy %g" % train_accuracy + ',    cross entropy %g' % cross_ent)
            train_accuracy_list.append(train_accuracy)
print('it took ' + str(np.round((time() - train_time0) / 60, 2)) + ' minutes to train network')
plt.plot(train_accuracy_list)

#
#                               TEST THE TRAINED NETWORK
#

test_prediction = []
prediction = tf.argmax(outputNeurons, 1)  # translating the prediction from one-hot to int
true_labels = np.argmax(test_labels, 1)
test_data = [resize(np.array(cube), (8, 28, 28)).flatten() for cube in test_data]
batch_size = 1
test_time0 = time()
for batch in range(len(test_labels) // batch_size):
    offset = (batch * batch_size) % len(train_data)
    batch_data = test_data[offset:(offset + batch_size)]
    batch_labels = test_labels[offset:(offset + batch_size)]
    test_prediction.append(prediction.eval(feed_dict={flat_cube: batch_data, y_: batch_labels, keep_prob: 1.0})[0])
t1 = time()
print('time to classify ' + str(len(test_labels)) + ' test data = ' + str(np.round(t1 - test_time0, 2)) + ' seconds')
print(str(np.round(len(test_labels) / (t1 - test_time0), 2)) + ' blobs per second labeled')
print(classification_report(test_prediction, true_labels))

