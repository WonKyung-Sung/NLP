#-*- coding: utf-8 -*-

import tensorflow as tf

class TextCnn_char(tf.keras.Model):
    def __init__(self, max_len_voca= 600):
        super(TextCnn_char, self).__init__()

        self.embeding_c = tf.keras.layers.Embedding(
            max_len_voca, 64, embeddings_initializer='uniform',
            embeddings_regularizer=None, activity_regularizer=None,
            embeddings_constraint=None, mask_zero=False, input_length=None)

        self.conv1 = tf.keras.layers.Conv2D(filters= 150, kernel_size = [3, 64], strides=(1, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.conv2_1 = tf.keras.layers.Conv2D(filters= 200, kernel_size = [2, 150], strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.conv2_2 = tf.keras.layers.Conv2D(filters= 200, kernel_size = [3, 150], strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.conv2_3 = tf.keras.layers.Conv2D(filters= 200, kernel_size = [4, 150], strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.conv2_4 = tf.keras.layers.Conv2D(filters= 200, kernel_size = [5, 150], strides=(1, 1), activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.1))
        self.d2 = tf.keras.layers.Dense(2, activation='softmax')
        

    def call(self, x):
        x = self.embeding_c(x)

        x = tf.expand_dims(x, 3)
        x = self.conv1(x)

        x = tf.transpose(x, perm=[0, 1, 3, 2])

        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)

        x2_1 = tf.nn.max_pool2d(x2_1, ksize=(1, (x2_1.shape[1] - x2_1.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')
        x2_2 = tf.nn.max_pool2d(x2_2, ksize=(1, (x2_2.shape[1] - x2_2.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')     
        x2_3 = tf.nn.max_pool2d(x2_3, ksize=(1, (x2_3.shape[1] - x2_3.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')
        x2_4 = tf.nn.max_pool2d(x2_4, ksize=(1, (x2_4.shape[1] - x2_4.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')

        x2 = tf.concat([x2_1,x2_2,x2_3,x2_4], axis=1)

        x2 = self.flatten(x2)
        x2 = self.d1(x2)
        return self.d2(x2)

class TextCnn_syllable(tf.keras.Model):
    def __init__(self, max_len_voca= 10000):
        super(TextCnn_syllable, self).__init__()

        self.embeding_c = tf.keras.layers.Embedding(
            max_len_voca, 200, embeddings_initializer='uniform',
            embeddings_regularizer=None, activity_regularizer=None,
            embeddings_constraint=None, mask_zero=False, input_length=None)
        self.conv2_1 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [2, 200], strides=(1, 1), activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [3, 200], strides=(1, 1), activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [4, 200], strides=(1, 1), activation='relu')
        self.conv2_4 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [5, 200], strides=(1, 1), activation='relu')
        
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(2, activation='softmax')
        
    def call(self, x):
        x = self.embeding_c(x)

        x = tf.expand_dims(x, 3)

        x2_1 = self.conv2_1(x)
        x2_2 = self.conv2_2(x)
        x2_3 = self.conv2_3(x)
        x2_4 = self.conv2_4(x)

        x2_1 = tf.nn.max_pool2d(x2_1, ksize=(1, (x2_1.shape[1] - x2_1.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')
        x2_2 = tf.nn.max_pool2d(x2_2, ksize=(1, (x2_2.shape[1] - x2_2.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')     
        x2_3 = tf.nn.max_pool2d(x2_3, ksize=(1, (x2_3.shape[1] - x2_3.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')
        x2_4 = tf.nn.max_pool2d(x2_4, ksize=(1, (x2_4.shape[1] - x2_4.shape[2] +1), 1, 1), strides=(1, 1,1,1), padding='VALID')

        x2 = tf.concat([x2_1,x2_2,x2_3,x2_4], axis=1)

        x2 = self.flatten(x2)
        x2 = self.d1(x2)
        return self.d2(x2)
