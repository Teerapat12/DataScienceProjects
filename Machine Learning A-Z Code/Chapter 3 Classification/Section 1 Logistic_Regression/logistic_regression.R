# Logistic Regression

#Importing the dataset 
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[,3:5]


#Splitting the dataset into the Training set and Test set
#install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased,SplitRatio=0.75) #Ratio for the training set
training_set = subset(dataset,split==TRUE)
test_set = subset(dataset,split==FALSE)


# Feature Scaling
training_set[,1:2] = scale(training_set[,1:2])
test_set[,1:2] = scale(test_set[,1:2])

#Fitting Logistic Regression to the Training set
classifier = glm(formula = Purchased~.,
                 family = binomial,
                 data=training_set)

#Predicting the Test set results
prob_pred = predict(classifier,type='response',newdata=test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

#Making the Confusion Matrix
cm = table(test_set[,3], y_pred)