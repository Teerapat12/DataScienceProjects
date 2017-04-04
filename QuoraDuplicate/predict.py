# -*- coding: utf-8 -*-

import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#Preprocessing
dataset = pd.read_csv("train.csv")
dataset = dataset.head(50000)  #This computer can't handle the size T T
dataset = dataset.fillna("")    

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             max_features = 10000) 


train_data_q1 = vectorizer.fit_transform(dataset['question1'])
train_data_q2 = vectorizer.transform(dataset['question2'])

train_data_q1 = train_data_q1.toarray()
train_data_q2 = train_data_q2.toarray()

#cleaned_data = np.concatenate((train_data_q1,train_data_q2),axis=1)
cleaned_data = train_data_q1*train_data_q2

#Training
X_train, X_test, y_train, y_test = train_test_split(    cleaned_data, dataset['is_duplicate'], test_size=0.4, random_state=0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#F2 score = (2*precision*recall)/(precision+recall)
recall = float(cm[1][1])/(cm[0][1]+cm[1][1])
precision = float(cm[1][1])/(cm[0][1]+cm[1][0])
f2score = 2*recall*precision/(recall+precision)
accuracy = float(cm[1][1]+cm[0][0])/(cm[1][0]+cm[0][1]+cm[1][1]+cm[0][0])
