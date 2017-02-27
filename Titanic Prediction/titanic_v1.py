# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#Create a function that clean data
def clean_titanic(tdf):
    tdf = tdf.fillna(tdf.median())
    tdf['NameLength'] = tdf.apply(lambda row : len(row['Name']),axis=1)
    SexDummies = pd.get_dummies(tdf['Sex'])
    tdf['Gender'] = SexDummies.iloc[:,0]
    EmbarkedDummies = pd.get_dummies(tdf['Embarked'])
    tdf = pd.concat([tdf,EmbarkedDummies.iloc[:,[1,2]]],axis=1)
    
    
    tdf['Fare_cheap']=0
    tdf['Fare_average']=0
    tdf['Fare_costly']=0
    for index,row in tdf.iterrows():
        if(row['Fare']<30):
            tdf.set_value(index,'Fare_cheap',1)
        elif(row['Fare']<70):
            tdf.set_value(index,'Fare_average',1)
        else:
            tdf.set_value(index,'Fare_costly',1)
            
    tdf['Mr'] = 0
    tdf['Mrs'] = 0
    tdf['Miss'] = 0
    tdf['royalty'] = 0
    tdf['officer'] = 0
    
    tdf['Relative'] = tdf['SibSp']+tdf['Parch']
    #depending on the name, categorizing individuals
    for index,row in tdf.iterrows():
        name = row['Name']
        if 'Mr.' in name:
            tdf.set_value(index,'Mr',1)
        elif 'Mrs.' in name:
            tdf.set_value(index,'Mrs',1)
        elif 'Miss.' in name:
            tdf.set_value(index,'Miss',1)
        elif 'Lady' or 'Don' or 'Dona' or 'sir' or 'master' in name:
            tdf.set_value(index,'royalty',1)
        elif 'rev' in name:
            tdf.set_value(index,'officer',1)
            
    tdf['Kid'] = 0
    tdf['Teenage'] = 0
    tdf['YoungAdult'] = 0
    tdf['Adult'] = 0
    tdf['OldAdult'] = 0
    tdf['Old'] = 0
    for index,row in tdf.iterrows():
        if(row['Age']<10):
            tdf.set_value(index,'Kid',1)
        elif(row['Age']<18):
            tdf.set_value(index,'Teenage',1)
        elif(row['Age']<30):
            tdf.set_value(index,'YoungAdult',1)
        elif(row['Age']<45):
            tdf.set_value(index,'Adult',1)
        elif(row['Age']<60):
            tdf.set_value(index,'OldAdult',1)
        elif(row['Age']<80):
            tdf.set_value(index,'Old',1)
            
    tdf = tdf.drop('Embarked',1)
    tdf = tdf.drop('Cabin',1)
    tdf = tdf.drop('Ticket',1)
    tdf = tdf.drop('Name',1)
    tdf = tdf.drop('Sex',1)
    tdf = tdf.drop('Fare_costly',1)
    tdf = tdf.drop('officer',1)
    tdf = tdf.drop('Old',1)
    #Exclude the Old
    return tdf


#Use clean function on both train and test
train = clean_titanic(train)
test = clean_titanic(test)


#Train the model with data. First we split train set so that we can use it for cross validation
# Splitting the dataset into the Training set and Test set
allCol= np.arange(2,23)
selectCol = [2,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,21]
X = train.iloc[:,selectCol]
y = train.iloc[:,1]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

'''
# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 14, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
'''

'''
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C=3.8,gamma=0.0189,random_state = 0)
classifier.fit(X_train, y_train)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20,scoring='accuracy')
accuracies.mean()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [
              {'C': np.arange(3.86,4,0.01), 'kernel': ['rbf'],'shrinking':[True,False]}]
#parameters = [{'max_depth':[4,5,6,7,8,None]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           verbose=10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


import statsmodels.formula.api as sm
regressor_OLS = sm.OLS(endog =y_train, exog = X_train).fit()
regressor_OLS.summary()
accuracies.mean()
#-----------------------------Test training with every data in train-----------------------------
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C=5.137,gamma=0.02,random_state = 0)
classifier.fit(X, y)

#-------------------- Next is to predict test with the classifier-------------------------------
X_validation = test.iloc[:,[i-1 for i in selectCol]]
X_validation = sc.transform(X_validation)
y_pred = classifier.predict(X_validation)
#Convert our answer to csv
passengerId = test['PassengerId']

columns = ['PassengerId','Survived']
predict_df = pd.DataFrame(
    {'PassengerId': test['PassengerId'],
     'Survived': y_pred},  columns=columns)
predict_df.to_csv('titanic_v1_svm_backward.csv',index=False)
