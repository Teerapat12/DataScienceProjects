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
isMale = pd.get_dummies(X['Sex'])
isMale= isMale.iloc[:,0]
X['Sex'] =  isMale
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

# Applying Kernel PCA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

# Fitting Random Forest Classifier to the Training set
'''
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)
'''


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


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

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train.values
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test.values
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Kernel SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#-------------------------------------------------------------------------------------------#

#Do the same data preprocessing on test set
test = pd.read_csv('test.csv')
test['Relatives'] = test['Parch']+test['SibSp']
Xtest = test.iloc[:, [1,3,4,-1,-4]]
isMale = pd.get_dummies(Xtest['Sex'])
isMale= isMale.iloc[:,0]
Xtest['Sex'] =  isMale
Xtest = Xtest.fillna(Xtest.mean())

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Xtest = sc.fit_transform(Xtest)
    
# Applying Kernel PCA to test set
Xtest = kpca.transform(Xtest)

y_pred = classifier.predict(Xtest)

#Convert our answer to csv
passengerId = test['PassengerId']

columns = ['PassengerId','Survived']
predict_df = pd.DataFrame(
    {'PassengerId': test['PassengerId'],
     'Survived': y_pred},  columns=columns)
predict_df.to_csv('titanic_predictions.csv',index=False)

