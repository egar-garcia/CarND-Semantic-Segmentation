import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests

import scipy.misc
import re
from glob import glob
import numpy as np
from sklearn.utils import shuffle


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Using tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    vgg_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return vgg_input, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out
tests.test_load_vgg(load_vgg, tf)


def conv_1x1(name, input_layer, num_classes, kernel_size = 1):
    return tf.layers.conv2d(
        input_layer, num_classes, kernel_size = 4,
        padding = 'SAME', 
        kernel_initializer = tf.random_normal_initializer(stddev = 0.01),
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
        name = name)

def upsample(name, input_layer, num_classes, kernel_size = 4, strides = (2, 2)):
    return tf.layers.conv2d_transpose(
        input_layer, num_classes, kernel_size, strides = strides, padding = 'SAME', 
        kernel_initializer = tf.random_normal_initializer(stddev = 0.01), 
        kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3),
        name = name)

def skip_layer(name, input_layer, output_layer):
    return tf.add(input_layer, output_layer, name = name)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    layer7_1x1  = conv_1x1('layer7_1x1', vgg_layer7_out, num_classes)
    layer7_up   = upsample('layer7_up', layer7_1x1, num_classes)
    layer4_1x1  = conv_1x1('layer4_1x1', vgg_layer4_out, num_classes)
    layer4_skip = skip_layer('layer4_skip', layer7_up, layer4_1x1)
    layer4_up   = upsample('layer4_up', layer4_skip, num_classes)
    layer3_1x1  = conv_1x1('layer3_1x1', vgg_layer3_out, num_classes)
    layer3_skip = skip_layer('layer3_skip', layer4_up, layer3_1x1)
    layer3_up   = upsample('nn_last_layer', layer3_skip, num_classes, kernel_size = 16, strides = (8, 8))

    return layer3_up
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name = 'logits')
    labels = tf.reshape(correct_label, (-1, num_classes), name = 'labels')

    # Loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))

    # Training operation
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        sum_loss = 0.0
        count = 0
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict = {input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 0.0009})
            sum_loss += loss
            count += 1
        print("Avg Loss: {:.3f}".format(sum_loss / count))
tests.test_train_nn(train_nn)



def get_augmented_batch_function(data_folder, image_shape):
    """
    Loads the training sample and augments it by flipping images,
    and generates function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
    background_color = np.array([255, 0, 0])

    images = []
    gt_images = []

    for image_file in image_paths:
        gt_image_file = label_paths[os.path.basename(image_file)]

        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

        images.append(image)
        gt_images.append(gt_image)
        images.append(np.fliplr(image))
        gt_images.append(np.fliplr(gt_image))

    images, gt_images = shuffle(images, gt_images)

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        for batch_i in range(0, len(images), batch_size):
            yield images[batch_i:batch_i+batch_size], gt_images[batch_i:batch_i+batch_size]
    return get_batches_fn


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    model_path = './model/model'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 20
    batch_size = 2

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        #get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        print("Getting and augmenting training set ...")
        # Augmenting images to the trainning set (for better results)
        get_batches_fn = get_augmented_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        print("Training set retrieved")

        # Building NN using load_vgg, layers, and optimize function

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name = 'correct_label')
        learning_rate = tf.placeholder(tf.float32, name = 'learning_rate')

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Training NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label,
                 keep_prob, learning_rate)

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        print("Model has been saved in: %s" % save_path)

        # Saving inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # To apply the trained model to a video see the script: video.py


if __name__ == '__main__':
    run()
