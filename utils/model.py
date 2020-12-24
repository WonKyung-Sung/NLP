#-*- coding: utf-8 -*-

import tensorflow as tf

class TextCnn_char(tf.keras.Model):
    def __init__(self, max_len_voca= 600):
        super(TextCnn_char, self).__init__()
        self.embeding_c = tf.keras.layers.Embedding(
            max_len_voca, 20, embeddings_initializer='uniform',
            embeddings_regularizer=None, activity_regularizer=None,
            embeddings_constraint=None, mask_zero=False, input_length=None)

        self.conv1 = tf.keras.layers.Conv2D(filters= 150, kernel_size = [3, 20], strides=(1, 3), activation='relu')
        self.conv2_1 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [2, 150], strides=(1, 1), activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [3, 150], strides=(1, 1), activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [4, 150], strides=(1, 1), activation='relu')
        self.conv2_4 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [5, 150], strides=(1, 1), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
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

class TextCnn(tf.keras.Model):
    def __init__(self, max_len_voca= 10000,  run_eagerly=True):
        super(TextCnn, self).__init__()

        self.embeding_c = tf.keras.layers.Embedding(
            max_len_voca, 200, embeddings_initializer='uniform',
            embeddings_regularizer=None, activity_regularizer=None,
            embeddings_constraint=None, mask_zero=False, input_length=None)
        self.conv2_1 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [2, 200], strides=(1, 1), padding='VALID', activation='relu')
        self.conv2_2 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [3, 200], strides=(1, 1), padding='VALID', activation='relu')
        self.conv2_3 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [4, 200], strides=(1, 1), padding='VALID', activation='relu')
        self.conv2_4 = tf.keras.layers.Conv2D(filters= 64, kernel_size = [5, 200], strides=(1, 1), padding='VALID', activation='relu')
        
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(2, activation='softmax')
        
    def call(self, inputs):
        x = self.embeding_c(inputs)
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


class Bert_classification(tf.keras.Model):
    def __init__(self, model_name, dir_path, num_class):
        super(Bert_classification, self).__init__()
        from transformers import TFBertModel
        self.bert = TFBertModel.from_pretrained(model_name, cache_dir=dir_path)
        self.dropout=tf.keras.layers.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier=tf.keras.layers.Dense(num_class, kernel_initializer=tf.keras.initializers.TruncatedNormal(self.bert.config.initializer_range), name="classifier")
    def call(self, inputs, attention_mask=None, token_type_ids=None, training=False):
        outputs= self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output=outputs[1]
        pooled_output=self.dropout(pooled_output, training=training)
        logits=self.classifier(pooled_output)

        return logits  
    
    
