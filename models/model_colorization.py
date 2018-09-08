from base.base_model import BaseModel
import tensorflow as tf


class ColorModel(BaseModel):
    def __init__(self, data_loader, config):
        super(ColorModel, self).__init__(config)
        # Get the data_generators to make the joint of the inputs in the graph
        self.data_loader = data_loader
        # define some important variables
        self.x = None
        self.y = None
        self.inception_embed=None
        self.is_training = None
        self.loss = None
        self.optimizer = None
        self.train_step = None


        self.build_model()
        self.init_saver()

    def build_model(self):
        """

        :return:
        """
        
        """
        Helper Variables
        """
        self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(self.global_step_tensor + 1)
        self.global_epoch_tensor = tf.Variable(0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(self.global_epoch_tensor + 1)
        
        """
        Inputs to the network
        """
        with tf.variable_scope('inputs'):
            self.x, self.y, self.inception_embed = self.data_loader.get_inputs()
            self.is_training = tf.placeholder(tf.bool, name='Training_flag')
            
        tf.add_to_collection('inputs', self.x)
        tf.add_to_collection('inputs', self.y)
        tf.add_to_collection('inputs', self.inception_embed)
        tf.add_to_collection('inputs', self.is_training)

        """
        Network Architecture
        """

        with tf.variable_scope('network'):
            with tf.variable_scope('encoder'):
                encoder_output = tf.layers.conv2d(self.x, 64, 3, strides=2, activation=tf.nn.relu, padding='same')
                encoder_output = tf.layers.conv2d(encoder_output, 128, 3, strides=1, activation=tf.nn.relu, padding='same')
                encoder_output = tf.layers.conv2d(encoder_output, 128, 3, strides=2, activation=tf.nn.relu, padding='same')
                encoder_output = tf.layers.conv2d(encoder_output, 256, 3, strides=1, activation=tf.nn.relu, padding='same')
                encoder_output = tf.layers.conv2d(encoder_output, 256, 3, strides=2, activation=tf.nn.relu, padding='same')
                encoder_output = tf.layers.conv2d(encoder_output, 512, 3, strides=1, activation=tf.nn.relu, padding='same')
                encoder_output = tf.layers.conv2d(encoder_output, 512, 3, strides=1, activation=tf.nn.relu, padding='same')
                encoder_output = tf.layers.conv2d(encoder_output, 256, 3, strides=1, activation=tf.nn.relu, padding='same')

            with tf.variable_scope('fusion'):
                fusion_output = tf.tile(self.inception_embed,[1,32*32])
                fusion_output = tf.reshape(fusion_output,[-1,32, 32, 1000])
                fusion_output = tf.concat([encoder_output, fusion_output], axis=3) 
                fusion_output = tf.layers.conv2d(fusion_output, 256, 1, activation=tf.nn.relu)

            with tf.variable_scope('decoder'):
                decoder_output = tf.layers.conv2d(fusion_output, 128, 3, strides=1, activation=tf.nn.relu, padding='same')
                decoder_output = tf.image.resize_images(decoder_output, size=(64,64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                decoder_output = tf.layers.conv2d(decoder_output, 64, 3, strides=1, activation=tf.nn.relu, padding='same')
                decoder_output = tf.image.resize_images(decoder_output, size=(128,128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                decoder_output = tf.layers.conv2d(decoder_output, 32, 3, strides=1, activation=tf.nn.relu, padding='same')
                decoder_output = tf.layers.conv2d(decoder_output, 16, 3, strides=1, activation=tf.nn.relu, padding='same')
                decoder_output = tf.layers.conv2d(decoder_output, 2, 3, strides=1, activation=tf.nn.tanh, padding='same')

            
            with tf.variable_scope('out'):
                self.out = tf.image.resize_images(decoder_output, size=(256,256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                tf.add_to_collection('out', self.out)

        """
        Some operators for the training process

        """

        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(labels=self.y, predictions=self.out)

        with tf.variable_scope('train_step'):
            self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('train', self.loss)

    def init_saver(self):
        """
        initialize the tensorflow saver that will be used in saving the checkpoints.
        :return:
        """
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep, save_relative_paths=True)