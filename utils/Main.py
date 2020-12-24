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
        self._data = False
        self.model = False
        self.voca_dic = False
        self.max_len_voca = False
        self._max_len_char = False
        self.tokenizer = False
        self._version = False
        
    @property
    def data(self):
        return self._data
    @data.setter
    def data(self, new_data):
        if type(new_data) != pd.DataFrame:
            raise ValueError("DataFrame 형식이 필요합니다.")
        self._data = new_data
        
    @property
    def max_len_char(self):
        return self._max_len_char
    @max_len_char.setter
    def max_len_char(self, new_max_len_char):
        if type(new_max_len_char) != int:
            raise ValueError("int 형식이 필요합니다.")
        self._max_len_char = new_max_len_char
        
    def preprocess(self, data, input_col="input", label_col="label", version="char3"):
        '''
        [설명]
            학습에서 사용할 데이터의 전처리를 진행함 
            따라서 해당 프로세스는 input data와 label이 모두 존재해야 함 
            또한 형태소 분석에 따라 활용가능한 알고리즘이 다르기 때문에 주의가 필요
        
        [input]
            data: 입력 데이터 (DataFrame)
            input_col(str): DataFrame일 경우 형태소 분석이 진행 될 컬럼명을 입력
            label_col(str): label의 컬럼 명을 입력  

            version: 활용할 형태소 분석기 명을 선택 
                     ("char3", "char", "okt", "khaiii", "bert" or "syllable" 사용가능)

                     char3: 
                            초성 중성 종성 형태로 나누어 주는 형태소 분석기 
                            (TextCnn_char, TextCnn 사용가능)
                     char: 
                           자소로 나누어 주는 형태소 분석기 
                           (TextCnn 사용가능)
                     okt: 
                          konlpy의 okt 형태소 분석기 
                          (TextCnn 사용가능)
                     khaiii: 
                             kakao의 khaiii 형태소 분석기 
                             (TextCnn 사용가능)
                     bert: 
                           bert의 형태소 분석기 
                           (Bert_classification 사용가능)
                     syllable: 
                           음절기반으로 나누어 주는 형태소 분석기
                           (TextCnn 사용가능)
        '''
        
        if label_col not in data.keys():
            raise Exception('label 컬럼이 존재하지 않습니다.') 
        if input_col not in data.keys():
            raise Exception('input 컬럼이 존재하지 않습니다.')
        self._version = version
        self.tokenizer = eval("wk.sent2" + version)
        
        if self._version not in ["bert"]:
            try: 
                with open(parent_path + "/data/voca_dic_" + version + ".pickle", 'rb') as f:
                    self.voca_dic = pickle.load(f)
            except:
                print("""기존 사전이 존재하지 않습니다. \n현 데이터 기반으로 새로 생성합니다.""")

            data.loc[:, "preprocess"] = self.tokenizer(data[input_col])
            if self.voca_dic == False:
                self.voca_dic = wk.vocabulary_maker(data['preprocess'])

            self.max_len_voca = len(self.voca_dic)
            if self._max_len_char == False:
                self._max_len_char = max(data['preprocess'].map(len).max(), 6) +1
                
            data.loc[:, 'input_'] = data.preprocess.map(
                lambda x: np.pad(
                    np.array([self.voca_dic.get(word, 1) for word in x]),
                    (0,self._max_len_char-len(x)),
                    'constant', constant_values=(0,0))
            )
            data.input_ = data.input_.map(lambda x: x[:500])
        else: 
            self._max_len_char = data[input_col].astype("str").map(len).max() +1 
            data.loc[:, "input_"] = pd.Series(
                self.tokenizer(data[input_col], MAX_LEN=self._max_len_char)
            )
        if label_col != "label":
            data.loc[:, "label"] = data[input_col]
        self._data =data

    def _predict_preprocess(self, data):
        if type(data) == str:
            data = pd.DataFrame({"input":[data]})
        else:
            data =pd.DataFrame({"input":data})
        if self._version not in ["bert"]:
            data.loc[:, "preprocess"] = self.tokenizer(data["input"])
            # self._max_len_char = max(data['preprocess'].map(len).max(), 6) +1
            data.loc[:, 'input_'] = data.preprocess.map(lambda x: np.pad( 
               np.array([self.voca_dic.get(word, 1) for word in x]) ,(0,self._max_len_char-len(x)),
               'constant', constant_values=(0,0)))
        else: 
            self._max_len_char = data["input"].astype("str").map(len).max()
            data.loc[:, "input_"] = pd.Series(
                self.tokenizer(data["input"], MAX_LEN=self._max_len_char)
            )
        return data
        
    def predict(self, data, batch=512):
        data = self._predict_preprocess(data)
        data = np.stack(data.input_, axis=0)
        
        if data.shape[0] < batch:
            batch = data.shape[0]
        data = tf.data.Dataset.from_tensor_slices(data).batch(batch)

        return self.model.predict(data)

    def select_model(self, selected_model="TextCnn_char", scrach=True):
        '''
        [설명]
            선택된 모델을 로드
        [input]
            selected_model(str): 사용할 모델명을 선택
                                 ("TextCnn_char", "TextCnn", or "Bert_classification")
            scrach: 완전 처음부터 학습 할지 여부를 결정
                    (True or False)
                
        [예시]
        
        '''
        
        if selected_model == "TextCnn_char":
            if scrach:
                import Model as wkm
                self.model = wkm.TextCnn_char(max_len_voca = self.max_len_voca)
            else:
                self.model = tf.keras.models.load_model('./path/to/location')
        if selected_model == "TextCnn":
            if scrach:
                import Model as wkm
                self.model = wkm.TextCnn(max_len_voca = self.max_len_voca)
            else:
                self.model = tf.keras.models.load_model('./path/to/location')
        if selected_model == "Bert_classification":
            import Model as wkm
            self.model = wkm.Bert_classification(model_name='bert-base-multilingual-cased',
                            dir_path='bert_ckpt',
                            num_class=2)

    def fit(self, test_ratio = 0.1, batch_size = 512, EPOCHS = 1):
        '''
        [설명]
            선택된 모델과 입력된 데이터를 기반으로 모델을 학습 
            
        [input]
            test_ratio: 학습/테스트 사이즈 설정
            batch_size: 배치 사이즈 설정
            EPOCHS(int): 학습 횟수 
        '''
        
        
        # 학습 나누기 
        raw_y = self._data[["label"]].to_numpy()
        raw_x = np.stack(self._data.input_, axis=0)
        x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y,  test_size = test_ratio,random_state=43)
        train_ds = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(x_train.shape[0] +1).batch(batch_size)        
#         train_ds = tf.data.Dataset.from_tensor_slices(
#             (x_train, y_train)).shuffle(x_train.shape[0] +1).batch(batch_size)
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

    
    def save(self, output_path="./"):
        '''
        해당 분석 프로세스에 필요한 자원들을 저장
        output_path(str): 분석 프로세스를 저장할 경로 위치
        '''
        
        if not os.path.isdir(output_path +"model/"): 
            os.makedirs(output_path +"model/")
        self.model.save(output_path + "model/model.h5")
        if not os.path.isdir(output_path +"class_info/"): 
            os.makedirs(output_path +"class_info/")
            
        pickle.dump(self.voca_dic, open(output_path +"class_info/"+"voca_dic.pickle", "wb"))
        pickle.dump(self.max_len_voca, open(output_path +"class_info/"+"max_len_voca.pickle", "wb"))
        pickle.dump(self._max_len_char, open(output_path +"class_info/"+"max_len_char.pickle", "wb"))
        pickle.dump(self.tokenizer, open(output_path +"class_info/"+"tokenizer.pickle", "wb"))
        pickle.dump(self._version, open(output_path +"class_info/"+"version.pickle", "wb"))
        
    def load(self, output_path):
        '''
        해당 분석 프로세스에 필요한 자원들을 로드
        output_path(str): 분석 프로세스가 저장되어 있는 경로 위치
        '''
        self.model = tf.keras.models.load_model(output_path + "model/model.h5")
        self.voca_dic = pickle.load(open(output_path +"class_info/"+"voca_dic.pickle", "rb"))
        self.max_len_voca = pickle.load(open(output_path +"class_info/"+"max_len_voca.pickle", "rb"))
        self._max_len_char = pickle.load(open(output_path +"class_info/"+"max_len_char.pickle", "rb"))
        self.tokenizer = pickle.load(open(output_path +"class_info/"+"tokenizer.pickle", "rb"))
        self._version = pickle.load(open(output_path +"class_info/"+"version.pickle", "rb"))
        
    def lime(self, text):
        '''
        [설명]
            현재 학습된 모델을 기반으로 lime의 텍스트 분석을 실행
        
        [input]
            text (str): 분석하기 위한 텍스트를 입력
            
        [예시]
        '''
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
        
