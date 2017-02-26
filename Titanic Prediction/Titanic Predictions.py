# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')

#Data preprocessing
dataset['Relatives'] = dataset['Parch']+dataset['SibSp']
X = dataset.iloc[:, [2,4,5,-1,-4]]
y = dataset.iloc[:, 1]

#Change Age from string to number so that it is easier to classify
X['Sex'][X['Sex']=='male'] = 1
X['Sex'][X['Sex']=='female'] = 0
#Fill null with the mean of each column
X = X.fillna(X.mean())

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classifier to the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)


'''
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''

'''
# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(random_state=0)
classifier.fit(X_train,y_train)
'''

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#-------------------------------------------------------------------------------------------#

#Suppose we choose Random Forest as our classifier
# Importing the dataset
dataset = pd.read_csv('train.csv')
#Data preprocessing for real model
dataset['Relatives'] = dataset['Parch']+dataset['SibSp']
X = dataset.iloc[:, [2,4,5,-1,-4]]
y = dataset.iloc[:,1]
#Change Age from string to number so that it is easier to classify
X['Sex'][X['Sex']=='male'] = 1
X['Sex'][X['Sex']=='female'] = 0
#Fill null with the mean of each column
X = X.fillna(X.mean())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


#Fit model to data
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(X,y)

#Do the same data preprocessing on test set
test = pd.read_csv('test.csv')
test['Relatives'] = test['Parch']+test['SibSp']
Xtest = test.iloc[:, [1,3,4,-1,-4]]
Xtest['Sex'][Xtest['Sex']=='male'] = 1
Xtest['Sex'][Xtest['Sex']=='female'] = 0
Xtest = Xtest.fillna(Xtest.mean())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtest = sc.fit_transform(Xtest)

y_pred = classifier.predict(Xtest)

#Convert our answer to csv
passengerId = test['PassengerId']

columns = ['PassengerId','Survived']
predict_df = pd.DataFrame(
    {'PassengerId': test['PassengerId'],
     'Survived': y_pred},  columns=columns)
predict_df.to_csv('titanic_predictions.csv',index=False)

