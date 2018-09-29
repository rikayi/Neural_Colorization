"""Input pipeline for the dataset.

"""
import os
import sys

sys.path.extend(['..'])

import numpy as np
import tensorflow as tf
from tensorflow.image import rgb_to_grayscale, grayscale_to_rgb

from utils.utils import get_args
from utils.config import process_config
from utils.imutils import check_image, rgb_to_lab



class DataLoader:
    def __init__(self, config):
        self.config = config

        data_dir = os.path.join('..','data', 'Dataset')
        train_dir = os.path.join(data_dir, 'Train')
        test_dir = os.path.join(data_dir, 'Test')

        # Get the file names from the train and dev sets
        self.train_filenames = np.array([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')])
        self.test_filenames  = np.array([os.path.join(test_dir, f)  for f in os.listdir(test_dir)  if f.endswith('.jpg')])

        
        # Define datasets sizes
        self.train_size = self.train_filenames.shape[0]
        self.test_size = self.test_filenames.shape[0]

        # Define number of iterations per epoch
        self.num_iterations_train = (self.train_size + self.config.batch_size - 1) // self.config.batch_size
        self.num_iterations_test  = (self.test_size  + self.config.batch_size - 1) // self.config.batch_size

        self.features_placeholder = None

        self.dataset = None
        self.iterator = None
        self.init_iterator_op = None
        self.next_batch = None

        #load embeddings 
        
        self.train_embeds=np.load('../../input/data/train_embeds.npy')
        self.test_embeds=np.load('../../input/data/test_embeds.npy')

        self._build_dataset_api()


    @staticmethod
    def _parse_function(filename, embed, size):
        """Obtain the image from the filename (for both training and validation).

        The following operations are applied:
            - Decode the image from jpeg format
            - Convert to float and to range [0, 1]
        """
        image_string = tf.read_file(filename)

        # Don't use tf.image.decode_image, or the output shape will be undefined
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)

        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)

        resized_image = tf.image.resize_images(image, [size, size])

        return resized_image,embed


    @staticmethod
    def _train_preprocess(image, embed, use_random_flip, mode='train'):
        """Image preprocessing for training.

        Apply the following operations:
            - Horizontally flip the image with probability 1/2
        """
        if mode == 'train':
            if use_random_flip:
                image = tf.image.random_flip_left_right(image)

        # convert image to LAB space and divide into features(L channel) and labels(AB channels)
        lab=rgb_to_lab(image)
        image=lab[:,:,0,None]
        label=lab[:,:,1:]/128


        return image, label, embed


    def _build_dataset_api(self):
        with tf.device('/cpu:0'):
            self.features_placeholder = tf.placeholder(tf.string, [None, ])
            self.mode_placeholder = tf.placeholder(tf.string, shape=())
            self.embed_placeholder= tf.placeholder(tf.float32,[None,1000])

            # Create a Dataset serving batches of images and labels
            # We don't repeat for multiple epochs because we always train and evaluate for one epoch
            parse_fn = lambda f, e: self._parse_function(f, e, self.config.image_size)
            train_fn = lambda f, e: self._train_preprocess(f, e, self.config.use_random_flip, self.mode_placeholder)

            self.dataset = (tf.data.Dataset.from_tensor_slices(
                    (self.features_placeholder, self.embed_placeholder)
                )
                .map(parse_fn, num_parallel_calls=self.config.num_parallel_calls)
                .map(train_fn, num_parallel_calls=self.config.num_parallel_calls)
                .batch(self.config.batch_size)
                .prefetch(1)  # make sure you always have one batch ready to serve
            )

            # Create reinitializable iterator from dataset
            self.iterator = self.dataset.make_initializable_iterator()

            self.iterator_init_op = self.iterator.initializer

            self.next_batch = self.iterator.get_next()

    def initialize(self, sess, mode='train'):
        if mode == 'train':
            idx = np.array(range(self.train_size))
            np.random.shuffle(idx)

            self.train_filenames = self.train_filenames[idx]
            self.train_embeds    = self.train_embeds[idx]

            sess.run(self.iterator_init_op, feed_dict={self.features_placeholder: self.train_filenames,
                                                   self.embed_placeholder: self.train_embeds,
                                                   self.mode_placeholder: mode})
        
        else:
            sess.run(self.iterator_init_op, feed_dict={self.features_placeholder: self.test_filenames,
                                                   self.embed_placeholder: self.test_embeds,
                                                   self.mode_placeholder: mode})


    def get_inputs(self):
        return self.next_batch


def main(config):
    """
    Function to test from console
    :param config:
    :return:
    """
    tf.reset_default_graph()

    sess = tf.Session()


    data_loader = DataLoader(config)
    images, labels, embeds = data_loader.get_inputs()
    print('Train')

    data_loader.initialize(sess, mode='train')

    out_im, out_l, out_e = sess.run([images, labels, embeds])

    print( out_im[0])
    print( out_l[0])
    print( out_e[0].shape)

    
    print('Test')
    data_loader.initialize(sess, mode='test')

    out_im, out_l, out_e = sess.run([images, labels, embeds])

    print(out_im.shape, out_im.dtype)
    print(out_l.shape, out_l.dtype)
    print(out_e.shape, out_e.dtype)


if __name__ == '__main__':
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
        main(config)

    except Exception as e:
        print('Missing or invalid arguments %s' % e)
