#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:43:02 2018

@author: sidharthdugar
"""

#importing libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#Getting data
train_data1 = pd.read_csv("train.txt",delimiter="__label__1",names=["Labels","Description"])
train_data1 = train_data1.dropna(subset=['Description'])
train_data1["Labels"] = train_data1["Labels"].fillna('__label__1')
train_data2 = pd.read_csv("train.txt",delimiter="__label__2",names=["Labels","Description"])
train_data2 = train_data2.dropna(subset=['Description'])
train_data2["Labels"] = train_data2["Labels"].fillna('__label__2')

train_data = train_data1.append(train_data2,ignore_index=True)

test_data = pd.read_csv("test.txt",names=["Description"],index_col=False,sep="\n")

#Pre-processing the train data
corpus = []
for i in range(0, 10000):
    review = re.sub('[^a-zA-Z]', ' ', train_data['Description'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 3500)
X = cv.fit_transform(corpus).toarray()
train_data = pd.get_dummies(train_data,columns=["Labels"])
del train_data['Labels___label__2']
y = train_data.iloc[:, 1].values

#Training the model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier(LogisticRegression(),n_estimators=500)
classifier.fit(X,y)

#preparing the testdata
corpus1 = []
for i in range(20000):
    review = re.sub('[^a-zA-Z]', ' ', test_data['Description'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus1.append(review)
    
X_test = cv.transform(corpus1).toarray()

#Cross-Validating for finding out better classifier
from sklearn.cross_validation import cross_val_score
accuracies = cross_val_score(classifier,X,y,n_jobs = -1,cv=10)
accuracies.mean()

#Predicting
y_pred = classifier.predict(X_test)

y_pred=pd.DataFrame(y_pred,columns=["Labels"])

y_pred = y_pred["Labels"].map({True:"__label__1",False:"__label__2"})
y_pred.to_csv('y_pred3.txt',index=False)

    