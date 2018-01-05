import tensorflow as tf
import numpy as np
from scipy.misc import imread
import random
import os
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
import tkinter as tk

class batch_handler():
    """Reads the image files and provides batches of subimages for the Neural Net to train on"""

    def __init__(self):
        self.data_dir = './'
        self.points = pd.read_csv(self._data_path('coords-clean.csv'))
        self.counts = pd.read_csv(self._data_path('train-clean.csv'))

        #Perhaps there is a validation.csv that contains a list of picture ids to use as the validation set
        #If not, randomly pick 1/10 of the pictures and create a csv containing those ids
        try:
            self.validation_ids = pd.read_csv(self._data_path('validation.csv')).as_matrix()[:,0]
        except:
            ids = np.array(self.counts.index).copy()
            np.random.shuffle(ids)
            self.validation_ids = ids[:int(0.1 * len(ids))]
            self.validation_ids.sort()
            pd.Series(self.validation_ids).to_csv(self._data_path('validation.csv'), index=False)
            print('WARNING: no validation set found.')

        self.training_ids = np.array(list(set(self.counts.index) - set(self.validation_ids)))
        self.training_ids.sort()

        self.shuffled_training_ids = self.training_ids.copy()
        random.shuffle(self.shuffled_training_ids)

    def _data_path(self, path):
        return '{}/{}'.format(self.data_dir, path)            

    def get_image(self, tid, img_type='Train'):
        """Reads an image

        Args:
            tid: image id number to read
            img_type: either 'Train' or 'Test'

        Returns:
            An mpimg image file of the picture
        """
        if (img_type not in ['Train', 'Test']):
            raise ValueError('get_image called with unknown type')
        img_path = '{}/{}/{}.jpg'.format(self.data_dir, img_type, tid)
        img = mpimg.imread(img_path)
        if img is None:
            return None
        return img

    def training_points(self, tid):
        """Returns the coordinates of all points in an image

        Args:
            tid: image id number to read

        Returns:
            numpy matrix with columns 'cls', 'row', 'col'
        """
        img_path = '{}/Train/{}.jpg'.format(self.data_dir, tid)
        with Image.open(img_path) as img:
            h, w = img.size
        points_in_rect = (self.points.tid == tid) & \
                         (0 <= self.points.row) & \
                         (self.points.row < h) & \
                         (0 <= self.points.col) & \
                         (self.points.col < w)
        points = self.points[points_in_rect] \
                     .as_matrix(['cls', 'row', 'col'])
        return points

    def _extract_blanks(self, num_pictures, window_diam):
        """Extracts subimages from the pictures with _NO_ sea lions 

        Args:
            num_pictures: number of pictures to extract
            window_diam: the edge length of the square subimages to extract

        Returns:
            numpy matrix of size (num_pictures, window_diam, window_diam, 3)
                where (i, :, :, :) contains the ith subimage
        """
        pics = np.zeros((num_pictures, window_diam, window_diam, 3), dtype='uint8')
        counter = 0

        for tid in shuffled_training_ids:
            img = self.get_image(tid)
            coords = self.training_points(tid)

            maxrow = (img.shape[0] // window_diam) - 1
            maxcol = (img.shape[1] // window_diam) - 1

            mask = np.ones((maxrow, maxcol), dtype='bool')
            
            for c in (coords[:, 1:] // window_diam):
                if (c[0] < maxrow and c[1] < maxrow):
                    mask[c[0], c[1]] = False

            for b in np.argwhere(mask):
                if ((b[0] % 2) and (b[1] % 2)):
                    r1, c1 = b[0] * window_diam, b[1] * window_diam
                    pics[counter] = img[r1:r1+window_diam, c1:c1+window_diam, :]
                    counter += 1
                    if (counter >= num_pictures):
                         break
            if (counter >= num_pictures):
                break
        return pics

    def _extract_mix(self, num_vacant, num_pictures, max_per_window, window_diam=224, progress_indicator=True):
        """Extracts a mix of subimages

        Args:
            num_vacant: number of empty subimages to include in the mix
            num_pictures: number of pictures of each class of sea lion to include in the mix
            max_per_window: maximum number of each class that the NN can detect in a subimage
            window_diam: the edge length of the square subimages to extract
            progress_indicator: set True to print progress statements to the console

        Returns: (pics, labels)
            pics: numpy array containing all subimages, each with size (window_diam x window_diam x 3) 
            labels: numpy array containing the labels for each image
        """
        num_total = num_vacant + (5 * num_pictures)
        label_length = (1 + (max_per_window * 5))
        pics = np.zeros((num_total, window_diam, window_diam, 3), dtype='uint8')
        labels = np.zeros((num_total, label_length), dtype='float32')
        pics[:num_vacant] = self._extract_blanks(num_vacant, window_diam)
        labels[:num_vacant, 0] = 1

        intra_picture_class_counter = np.zeros(5, dtype=int)
        
        window_rad = window_diam // 2
        total_counter = num_vacant

        max_from_image = 30
        mfi_counter = 0

        for class_num in range(5):
            class_counter = 0
            if (progress_indicator == True):
                print('Processing class {}'.format(class_num))
            for tid in shuffled_training_ids:
                if (progress_indicator == True):
                    print('Processing {} of class {}, finished {}/{}'.format(tid, class_num, class_counter, num_pictures))
                img = self.get_image(tid)
                coords = self.training_points(tid)
                maxrow = img.shape[0] - window_rad - 1
                maxcol = img.shape[1] - window_rad - 1
                mfi_counter = 0
                if (coords.shape[0] >= max_from_image):
                    np.random.shuffle(coords)

                for c in coords:
                    if (c[0] == class_num and (window_rad <= c[1] <= maxrow) and (window_rad <= c[2] <= maxcol)):
                        pics[total_counter] = img[c[1] - window_rad: c[1]-window_rad + window_diam, c[2]-window_rad: c[2]-window_rad + window_diam, :]
                        intra_picture_class_counter = np.zeros(5, dtype=int)
                        for i in np.argwhere((c[1] - window_rad <= coords[:,1]) &
                                             (coords[:,1] <= c[1]+window_rad) &
                                             (c[2] - window_rad <= coords[:,2]) &
                                             (coords[:,2] <= c[2]+window_rad)):
                            intra_picture_class_counter[coords[i[0], 0]] += 1 
                            
                        for i in range(5):
                            num_to_change = min(max_per_window, intra_picture_class_counter[i])
                            labels[total_counter, (max_per_window * i) + 1: (max_per_window * i) + 1 + num_to_change] = 1
                            
                        #labels[total_counter] = labels[total_counter] / labels[total_counter].sum() #changes it into a probability distribution
                        total_counter += 1
                        class_counter += 1
                        mfi_counter += 1
                    if ((class_counter >= num_pictures) or (mfi_counter >= max_from_image)):
                        break
                if (class_counter >= num_pictures):
                    break
        return pics, labels                       

    def initialize_mixed_training_set(self, num_vacant, num_per_class, max_per_window, window_diam=224):
        """Initializes the following attributes of batch_handler to prepare for training:
            training_data: a collection of subimages with num_vacant empty pictures
                and num_per_class pictures centered on each type of sea lion
            training_labels:
            randomized_order: a shuffled list of indices of size len(training_data)
            current_counter: 0
        """
        self.training_data, self.training_labels = self._extract_mix(num_vacant, num_per_class, max_per_window, progress_indicator=True)
        self.randomized_order = list(range(len(self.training_data)))
        random.shuffle(self.randomized_order)
        self.current_counter = 0        
        

    def serve_batch(self, batch_size):
        """Returns a training batch of size batch_size"""
        self.current_counter += batch_size
        if (self.current_counter >= len(self.randomized_order)):
            self.current_counter = batch_size
            print('going back to the beginning of the data')
        return self.training_data[self.randomized_order[self.current_counter - batch_size: self.current_counter]], self.training_labels[self.randomized_order[self.current_counter - batch_size: self.current_counter]]


#----------------------The VGG16 architecture was downloaded from---------------------
#----------------------Davi Frossard's website. The header is below-------------------

########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

class vgg16:
    def __init__(self, imgs, labels=None, weights=None, sess=None):
        self.imgs = imgs
        self.labels = labels
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        # this takes the image and subtracts a number from each of the RGB channels, effectively making the mean equal to 0
        # it stores the result in the local variable 'images'
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        # sets up a first convolutional layer with 64 channels, using a 3x3(x3) kernel
        # conv -> conv + bias -> relu -> saved into self.conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        print('Loading weights from {}'.format(weight_file))
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print (i, k, np.shape(weights[k]))
            sess.run(self.parameters[i].assign(weights[k]))

    def reset_final_layer(self, max_per_window):
        num_classes = 1 + (max_per_window * 5)
        with tf.name_scope('fc3_new') as scope:
            fc3w_new = tf.Variable(tf.truncated_normal([4096, num_classes],
                                                           dtype=tf.float32,
                                                           stddev=1e-1), name='weights')
            fc3b_new = tf.Variable(tf.constant(1.0, shape=[num_classes], dtype=tf.float32), trainable=True, name='biases')
            self.fc31_new = tf.nn.bias_add(tf.matmul(self.fc2, fc3w_new), fc3b_new)
            self.parameters = self.parameters[:-2] + [fc3w_new, fc3b_new]
            self.sigmoid_output = tf.nn.sigmoid(self.fc31_new)
            sess.run(tf.variables_initializer([self.parameters[-2], self.parameters[-1]]))
        

class visualizer():

    def __init__(self, sess, vgg, bh, max_per_window = 40,
                tk_height=600, tk_width=620, img_height=400, img_width=600,
                default_pause=50, filled_pause=200, sigmoid_threshold=0.4):
        self.default_pause = default_pause
        self.filled_pause = filled_pause
        self.tk_height = tk_height
        self.tk_width = tk_width
        self.img_height = img_height
        self.img_width = img_width
        self.sigmoid_threshold = sigmoid_threshold
        self.root = tk.Tk()
        self.f = tk.Frame(self.root, height=self.tk_height, width=self.tk_width)
        self.f.pack()
        self.canv = tk.Canvas(self.f, height=self.tk_height, width=self.tk_width)
        self.canv.pack()
        self.counter = 0
        self.bh = bh
        self.vgg = vgg
        self.sess = sess
        self.sliding_window_total_count = np.zeros(5)
        self.max_per_window = max_per_window
        

    def one_hot_to_numbers(self, y):
        opt = np.zeros(5)
        for i in range(5):
            opt[i] = y[(self.max_per_window * i) + 1: (self.max_per_window * (i+1)) + 1].sum()
        return opt

    def done_sliding_window(self, coords):
        self.canv.delete('window')
        self.canv.delete('sliding_predictions')
        picture_tally_truth = np.zeros(5)
        U, V = np.unique(coords[:, 0], return_counts=True)
        for i in range(len(U)):
            picture_tally_truth[U[i]] = V[i]
        self.canv.create_text(10, self.img_height + 10, text='Prediction: {}'.format(self.sliding_window_total_count), anchor=tk.NW)
        self.canv.create_text(10, self.img_height + 40, text='     Truth: {}'.format(picture_tally_truth), anchor=tk.NW)
        

    def update_sliding_window(self, img, x, y, stride, coords, window_diam=224):
        #amount_to_wait is the number of milliseconds that the screen pauses if it finds a square with sea lions in it
        amount_to_wait = self.filled_pause
        
        x_scale_factor = img.shape[1] / self.img_width
        y_scale_factor = img.shape[0] / self.img_height
        window = img[y:y+window_diam, x:x+window_diam]
        sig_output = self.sess.run(self.vgg.sigmoid_output, feed_dict={self.vgg.imgs:[window]})[0]
        square_tally = self.one_hot_to_numbers(sig_output)

        nwc_x = 10 + ((600 * x) // img.shape[1])
        nwc_y = 10 + ((400 * y) // img.shape[0])

        scaled_w_x = ((600 * window_diam) // img.shape[1])
        scaled_w_y = ((400 * window_diam) // img.shape[0])
        scaled_wd = window_diam // 10
    
        
        self.canv.delete('window')
        self.canv.delete('sliding_predictions')

        self.canv.create_line(nwc_x, nwc_y, nwc_x+scaled_w_x, nwc_y, fill='yellow', tag='window')
        self.canv.create_line(nwc_x+scaled_w_x, nwc_y, nwc_x+scaled_w_x, nwc_y+scaled_w_y, fill='yellow', tag='window')
        self.canv.create_line(nwc_x+scaled_w_x, nwc_y+scaled_w_y, nwc_x, nwc_y+scaled_w_y, fill='yellow', tag='window')
        self.canv.create_line(nwc_x, nwc_y+scaled_w_y, nwc_x, nwc_y, fill='yellow', tag='window')

        self.canv.create_text(10, self.img_height + 10, text='        Window Prediction: {}'.format(square_tally),
                                anchor=tk.NW, tag='sliding_predictions')
        square_tally[square_tally < self.sigmoid_threshold] = 0
        self.canv.create_text(10, self.img_height + 30, text=' Cutoff Window Prediction: {}'.format(square_tally),
                                anchor=tk.NW, tag='sliding_predictions')
        self.sliding_window_total_count += square_tally
        self.canv.create_text(10, self.img_height + 50, text='Running Tally for Picture: {}'.format(self.sliding_window_total_count),
                                anchor=tk.NW, tag='sliding_predictions')

        
        if (square_tally.sum() == 0):
            amount_to_wait = 10

        x = x + stride
        if (x + window_diam >= img.shape[1]):
            y = y + stride
            x = 0
        if (y + window_diam < img.shape[0]):
            self.canv.after(amount_to_wait, lambda: self.update_sliding_window(img, x, y, stride, coords, window_diam))
        else:
            self.canv.after(amount_to_wait, lambda: self.done_sliding_window(coords))
                    

    def sliding_window(self, tid, show_truth=True, window_diam=224, stride=224, pause_time=50):
        self.canv.delete('all')
        img = bh.get_image(tid)
        coords = bh.training_points(tid)
        self.sliding_window_total_count = np.zeros(5)
        resized_img = Image.fromarray(img).resize((self.img_width, self.img_height))
        self.phot_img = ImageTk.PhotoImage(resized_img)
        self.canv.create_image(10, 10, image=self.phot_img, anchor=tk.NW)
        self.canv.after(pause_time, lambda: self.update_sliding_window(img, 0, 0, stride, coords, window_diam))


def train_cnn(sess, vgg, bh, num_steps=5000, learning_rate=0.002):
    """Trains the neural network
    
    Params:
        vgg: instance of vgg16 class, contains the NN architecture
        bh: instance of batch_handler class, serves batches of training data and labels
        num_steps: integer number of images to train on 
        learning_rate: learning rate for gradient descent
    """
    batch_size = 1
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=vgg.labels, logits=vgg.fc31_new, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    sess.run(tf.variables_initializer([global_step]))
    train_op = optimizer.minimize(loss, global_step=global_step)
    loss_summary = 0
    
    for step in range(num_steps):
        X, Y = bh.serve_batch(batch_size)
        _, loss_value, sig_output = sess.run([train_op, loss, vgg.sigmoid_output],
                                                feed_dict={vgg.imgs: X, vgg.labels: Y})
        print('On step {0}, loss value = {1:.5f}'.format(step, loss_value))
        loss_summary += loss_value
        if ((step + 1) % 100 == 0):
            print('Average loss over last 100 steps = {}'.format(loss_summary/100))
            loss_summary = 0
        


def test_cnn(sess, vgg, bh, pictures=None, threshold=0.6,
            sigmoid_threshold=0.4, window_diam=224, max_per_window=40,
            stride=224, pup_cap=True, adult_male_cap=True):
    """Tests the trained NN on the validation set
    
    Params:
        pictures: a list of training ids to process; if None, use the validation ids
        threshold:
        sigmoid_threshold:
        window_diam:
        max_per_window:
    """
    if pictures is None:
        pictures = bh.validation_ids

    window_tally = np.zeros(5)
    all_predictions = np.zeros((len(pictures), 5))

    #next two lines are grid specific
    sigmoid_grid = np.zeros(1+(max_per_window*5))
    window_tally_grid = np.zeros(5)
    grid_predictions = np.zeros((len(pictures), 10, 11, 5))
                                
    all_truths = np.zeros((len(pictures), 5))
    counter = 0
    
    for tid in pictures:
        print('\nAnalyzing picture {} of {}, tid={}'.format(counter+1, len(pictures), tid))
        img = bh.get_image(tid)
        coords = bh.training_points(tid)

        #calculate true counts
        U, V = np.unique(coords[:, 0], return_counts=True)
        for i in range(len(U)):
            all_truths[counter, U[i]] = V[i]

        #predict counts with a sliding window
        x = 0
        y = 0
        while(y + window_diam < img.shape[0]):
            window = img[y:y+window_diam, x:x+window_diam]

            #count the sea lions in the window, threshold it at 0.45, then add the result to all_predictions
            sig_output = sess.run(vgg.sigmoid_output, feed_dict={vgg.imgs:[window]})[0]

            #next two lines are grid specific
            sigmoid_grid = sig_output.copy()
            for a in range(10):
                sigmoid_grid[sigmoid_grid < (a * .1)] = 0
                for j in range(5):
                    window_tally_grid[j] = sigmoid_grid[(max_per_window * j) + 1: (max_per_window * (j+1)) + 1].sum()
                for b in range(11):
                    window_tally_grid[window_tally_grid < (b * .1)] = 0
                    grid_predictions[counter, a, b, :] += window_tally_grid
            
            sig_output[sig_output < sigmoid_threshold] = 0
            for j in range(5):
                window_tally[j] = sig_output[(max_per_window * j) + 1: (max_per_window * (j+1)) + 1].sum()
            window_tally[window_tally < threshold] = 0

            
            all_predictions[counter, :] += window_tally

            #move the window
            x = x + stride
            if (x + window_diam >= img.shape[1]):
                y = y + stride
                x = 0

        #Apply corrections to prevent counting seals as pups, and avoid counting rocks as adult males   
        if ((pup_cap is True) and (all_predictions[counter, 4] > all_predictions[counter, 2])):
            all_predictions[counter, 4] = all_predictions[counter, 2]
        if ((adult_male_cap is True) and (all_predictions[counter, 0] > 50)):
            all_predictions[counter, 0] = 50

        for a in range(10):
            for b in range(11):
                if ((pup_cap is True) and (grid_predictions[counter,a,b, 4] > grid_predictions[counter,a,b, 2])):
                    grid_predictions[counter,a,b, 4] = grid_predictions[counter,a,b, 2]
                if ((adult_male_cap is True) and (grid_predictions[counter,a,b, 0] > 50)):
                    grid_predictions[counter,a,b, 0] = 50                

        #Print the summary comparing predictions to truth
        error = all_predictions[counter, :] - all_truths[counter, :]
        print('Prediction: {}'.format(all_predictions[counter, :]))
        print('Truth: {}'.format(all_truths[counter, :]))
        print('Squared Error: {}'.format(error * error))
        counter += 1

    all_errors = all_predictions - all_truths
    overall_score = np.sqrt(((all_errors * all_errors).sum(axis=0)) / (len(pictures)))
    print('Overall score: mean of {0} is {1:.4f}'.format(overall_score, (overall_score.sum())/5))
    return all_predictions, grid_predictions, all_truths


def produce_results(sess, vgg, bh, pictures=None, threshold=0.6, sigmoid_threshold=0.4, window_diam=224,
                    max_per_window=40, stride=224, pup_cap=True):
    """Produces the results on the testset in the format required by kaggle
    
    Params:
        sess: instance of the tensorflow session
        vgg: instance of the vgg16 class
        bh: instance of the batchhandler class
        threshold: If less than threshold of any class are in a picture then zero of that class are reported
        sigmoid_threshold: on any subimage, if the sigmoid output is less than sigmoid_threshold, round down to 0
        window_diam: shape of the window
        max_per_window: number of each class that can be counted in the window
        stride: amount of stride for the sliding window
        pup_cap: when True, limit the number of pups by the number of adult females
    """

    window_tally = np.zeros(5)
    all_predictions = np.zeros((len(pictures), 5))
    picture_counter = 0
    
    for tid in pictures:
        print('\nAnalyzing picture {} of {}, tid={}'.format(picture_counter+1, len(pictures), tid))
        img = bh.get_image(tid, img_type='Test')

        #predict counts with a sliding window
        x = 0
        y = 0
        while(y + window_diam < img.shape[0]):
            window = img[y:y+window_diam, x:x+window_diam]

            #count the sea lions in the window, threshold it at 0.45, then add the result to all_predictions
            sig_output = sess.run(vgg.sigmoid_output, feed_dict={vgg.imgs:[window]})[0]
            
            sig_output[sig_output < sigmoid_threshold] = 0
            for j in range(5):
                window_tally[j] = sig_output[(max_per_window * j) + 1: (max_per_window * (j+1)) + 1].sum()
            window_tally[window_tally < threshold] = 0            
            all_predictions[picture_counter, :] += window_tally

            #move the window
            x = x + stride
            if (x + window_diam >= img.shape[1]):
                y = y + stride
                x = 0
                
        #Apply corrections to prevent counting seals as pups   
        if ((pup_cap is True) and (all_predictions[picture_counter, 4] > all_predictions[picture_counter, 2])):
            all_predictions[picture_counter, 4] = all_predictions[picture_counter, 2]
        picture_counter += 1

    print('Done. Saving...')
    ofname = '{}.csv'.format(pictures[len(pictures) - 1])
    np.savetxt(ofname, np.hstack((np.array(pictures).astype(int).reshape((-1, 1)), all_predictions)), header='test_id,adult_males,subadult_males,adult_females,juveniles,pups', fmt='%1.f, %1.f, %1.f, %1.f, %1.f, %1.f', comments='')
    return




if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    #The following option changes the architecture of the NN, it determines how many output bits to allocate for each class
    #   if you change this to a number other than 30, you cannot use the saved checkpoint
    max_per_window = 40
    
    #These options are for training:
    #   training_mode is True if you want the NN to train, false otherwise
    #   checkpoint_mode is True if you want to save your progress, false otherwise
    #   num_vacant is the number of pictures to put in the training set with 0 sea lions
    #   num_per_class is the number of pictures centered on sea lions (of each class) to put in the training set
    #       This means there will be num_vacant + 5(num_per_class) pictures total
    #   num_steps is the total number of batches to process during training
    training_mode = False
    restore_from_checkpoint_mode = True
    save_to_checkpoint_mode = False
    num_vacant = 60
    num_per_class = 60
    num_steps = 300
    learning_rate = 0.000001
    

    #Initialize the tensorflow session, create the computational graph, and change the final layer
    #If restore_from_checkpoint_mode is True, it will load the weights from the given checkpoint file
    #   otherwise, the weights for the last layer will be randomized
    #The placeholders imgs and labs hold the inputs (subimages and labels, respectively) to the NN
    #   Our labels will have one position for 'vacant', and max_per_window positions for each of the 5 classes
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labs = tf.placeholder(tf.float32, [None, (1 + (max_per_window * 5))])
    vgg = vgg16(imgs, labs, 'vgg16_weights.npz', sess)
    vgg.reset_final_layer(max_per_window)
    
    saver = tf.train.Saver()
    if (restore_from_checkpoint_mode is True):
        saver.restore(sess, 'final_weights_p01_max40.ckpt')

    #bh supplies the pictures.
    bh = batch_handler()
    if (training_mode is True):
        bh.initialize_mixed_training_set(num_vacant, num_per_class, max_per_window, window_diam=224)    
        train_cnn(sess, vgg, bh, num_steps, learning_rate=learning_rate)
    
    #save a checkpoint
    if (save_to_checkpoint_mode is True):
        print('Saving')
        saver.save(sess, os.path.join(os.getcwd(), 'final_weights_p01_max40.ckpt'))
        print('Done')

    #To run a sliding window visualization on tid 122:    
    #   v = visualizer(sess, vgg, bh)
    #   v.sliding_window(122)