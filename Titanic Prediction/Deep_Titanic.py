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

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first layer
classifier.add(Dense(output_dim = 3,init = 'uniform',activation='relu',input_dim= 5))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 3,init = 'uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation='sigmoid')) #Sufmax ? for more than one category

# Compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics= ['accuracy']) #categorical_crossentropy

# Fitting the ANN to the Training set
classifier.fit(X_train,y_train, batch_size=10, nb_epoch=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred>0.5

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#-------------------------------------------------------------------------------------------#

#Suppose we choose Deep Learning as our classifier
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

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first layer
classifier.add(Dense(output_dim = 2,init = 'uniform',activation='relu',input_dim= 5))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 2,init = 'uniform',activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation='sigmoid')) #Sufmax ? for more than one category

# Compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics= ['accuracy']) #categorical_crossentropy

# Fitting the ANN to the Training set
classifier.fit(X,y, batch_size=10, nb_epoch=100)

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
y_pred = y_pred>0.5
y_pred = y_pred*1

#Convert our answer to csv
passengerId = test['PassengerId']

columns = ['PassengerId','Survived']
predict_df = pd.DataFrame(
    {'PassengerId': test['PassengerId'],
     'Survived': y_pred.flatten()},  columns=columns)
predict_df.to_csv('titanic_predictions_ann.csv',index=False)

