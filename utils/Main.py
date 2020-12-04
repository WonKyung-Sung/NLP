#-*- coding: utf-8 -*-

import re
import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
sys.path.append("../")
parent_path = os.path.abspath('.')
import text_preprocess as wk
# import Model as WKM

class learning:
    def __init__(self):
        self.data = False
        self.model = False
        self.voca_dic = False
        self.max_len_voca = False
        self.max_len_char = False
        self.tokenizer = False
        
    def preprocess(self, data, input_col="input", label_col="label", version="char3"):
        if label_col not in data.keys():
            raise Exception('label 컬럼이 존재하지 않습니다.') 
        if input_col not in data.keys():
            raise Exception('input 컬럼이 존재하지 않습니다.')
            
        self.tokenizer = eval("wk.sent2" + version)
        try: 
            with open(parent_path + "/data/voca_dic_" + version + ".pickle", 'rb') as f:
                self.voca_dic = pickle.load(f)
        except:
            print("""기존 사전이 존재하지 않습니다. \n현 데이터 기반으로 새로 생성합니다.""")
        
        data["preprocess"] = self.tokenizer(data[input_col])
        if self.voca_dic == False:
            self.voca_dic = wk.vocabulary_maker(data['preprocess'])
        
        self.max_len_voca = len(self.voca_dic)
        print(len(self.voca_dic))
        self.max_len_char = data['preprocess'].map(len).max()
        
#         data['input_'] = data.preprocess.map(lambda x:
#                                              np.array(
#                                                  [self.voca_dic.get(word, 0) for word in x])
#                                             )
        data['input_'] = data.preprocess.map(lambda x: np.pad( 
            np.array([self.voca_dic.get(word, 0) for word in x]) ,(0,self.max_len_char-len(x)),
            'constant', constant_values=(0,0)))
        if label_col != "label":
            data["label"] = data[input_col]
        self.data =data

    def predict_preprocess(self, data):
        if type(data) == str:
            data = pd.DataFrame({"input":[data]})
        else:
            data =pd.DataFrame({"input":data})
            
        data["preprocess"] = self.tokenizer(data["input"])    
#         data['input_'] = data.preprocess.map(
#             lambda x: np.array(
#                 [self.voca_dic.get(word, 0) for word in x]
#             )
#         )
        self.max_len_char = data['preprocess'].map(len).max()
        data['input_'] = data.preprocess.map(lambda x: np.pad( 
           np.array([self.voca_dic.get(word, 0) for word in x]) ,(0,self.max_len_char-len(x)),
           'constant', constant_values=(0,0)))
        return data
        
    def predict(self, data):
        data = self.predict_preprocess(data)
        data = np.stack(data.input_, axis=0)
        data = tf.data.Dataset.from_tensor_slices(data).batch(512)
        return self.model.predict(data)

    def select_model(self, selected_model="TextCnn_char", scrach=True):
        if selected_model == "TextCnn_char":
            print(1)
            if scrach:
                import Model as wkm
                self.model = wkm.TextCnn_char(max_len_voca = self.max_len_voca)
            else:
                self.model = tf.keras.models.load_model('./path/to/location')
        if selected_model == "TextCnn_syllable":
            print(2)
            if scrach:
                import Model as wkm
                self.model = wkm.TextCnn_syllable(max_len_voca = self.max_len_voca)
            else:
                self.model = tf.keras.models.load_model('./path/to/location')

    def fit(self, test_ratio = 0.1, batch_size = 512, EPOCHS = 1):
        # 학습 나누기 
        raw_y = self.data[["label"]].to_numpy()
        raw_x = np.stack(self.data.input_, axis=0)
        x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y,  test_size = test_ratio,random_state=43)
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(x_train.shape[0] +1).batch(batch_size)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Nadam()
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        @tf.function
        def train_step(batch_input, labels):
            with tf.GradientTape() as tape:
                predictions = self.model(batch_input)
                loss = loss_object(labels, predictions)
                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                train_loss(loss)
                train_accuracy(labels, predictions)

        @tf.function
        def test_step(batch_input, labels):
            predictions = self.model(batch_input)
            t_loss = loss_object(labels, predictions)

            test_loss(t_loss)
            test_accuracy(labels, predictions)

        for epoch in range(EPOCHS):
            for batch_input, labels in train_ds:
                train_step(batch_input, labels)


            for test_batch_input, test_labels in test_ds:
                test_step(test_batch_input, test_labels)

            template = '에포크: {}, 손실: {}, 정확도: {}, 테스트 손실: {}, 테스트 정확도: {}'
            print (template.format(epoch+1,
                             train_loss.result(),
                             train_accuracy.result()*100,
                             test_loss.result(),
                             test_accuracy.result()*100))
            # model.save('path/to/location')

    def data(self):
        return self.data
    def model(self):
        return self.model
    def data(self):
        return self.voca_dic
    
    def save(self, output_path):
        if not os.path.isdir(output_path +"model/"): 
            os.makedirs(output_path +"model/")
        self.model.save(output_path + "model")
        if not os.path.isdir(output_path +"class_info/"): 
            os.makedirs(output_path +"class_info/")
            
        pickle.dump(self.voca_dic, open(output_path +"class_info/"+"voca_dic.pickle", "wb"))
        pickle.dump(self.max_len_voca, open(output_path +"class_info/"+"max_len_voca.pickle", "wb"))
        pickle.dump(self.max_len_char, open(output_path +"class_info/"+"max_len_char.pickle", "wb"))
        pickle.dump(self.tokenizer, open(output_path +"class_info/"+"tokenizer.pickle", "wb"))
        
    def load(self, output_path):
        self.model = tf.keras.models.load_model(output_path + "model")
        self.voca_dic = pickle.load(open(output_path +"class_info/"+"voca_dic.pickle", "rb"))
        self.max_len_voca = pickle.load(open(output_path +"class_info/"+"max_len_voca.pickle", "rb"))
        self.max_len_char = pickle.load(open(output_path +"class_info/"+"max_len_char.pickle", "rb"))
        self.tokenizer = pickle.load(open(output_path +"class_info/"+"tokenizer.pickle", "rb"))
        
    def lime(self, text):
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=["부정","긍정"])
        max_len = len(text.split(" "))
        explanation = explainer.explain_instance(text, classifier_fn = self.predict, labels=(1,), num_features=max_len)
        explanation.show_in_notebook(text=True)
        return explanation.as_list()
    
    def report(self, pred, labels, target_names=False):
        '''
        pred = [0, 0, 2, 2, 1]
        labels = [0, 1, 2, 2, 2]
        target_names = ['class 0', 'class 1', 'class 2']       
        '''
        if np.array(pred).shape[-1] == 2:
            pred = np.argmax(pred, 1)
        
        from sklearn.metrics import classification_report
        if target_names:
            print(classification_report(labels, pred, target_names=target_names))
        else:
            print(classification_report(labels, pred))
        
