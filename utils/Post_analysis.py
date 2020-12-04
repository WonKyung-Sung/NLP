#-*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def lime(text, predict):
    from lime.lime_text import LimeTextExplainer
    explainer = LimeTextExplainer(class_names=["부정","긍정"])
    max_len = len(text.split(" "))
    explanation = explainer.explain_instance(text, classifier_fn = predict, labels=(1,), num_features=max_len)
    explanation.show_in_notebook(text=True)
    return explanation.as_list()

def report(pred, labels, target_names=False):
    '''
    pred = [0, 0, 2, 2, 1] or array([[0.47921422, 0.5207857 ],[0.47803602, 0.52196395]])
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
