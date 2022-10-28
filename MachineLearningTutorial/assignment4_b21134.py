import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from math import pi,log

#reading the csv file as a dataframe
X=pd.read_csv('SteelPlateFaults-2class.csv').drop(columns=['Class'])
X_label = pd.read_csv('SteelPlateFaults-2class.csv')['Class']

#splitting data into train data and test data
[X_train, X_test, X_label_train, X_label_test] = train_test_split(X, X_label, test_size=0.3, random_state=42,shuffle=True)

#saving the train data and test data as csv files
X_train['Class'] = X_label_train
X_test['Class'] = X_label_test
X_train.to_csv('SteelPlateFaults-train.csv',index=False)
X_test.to_csv('SteelPlateFaults-test.csv',index=False)
X_train = X_train.drop(columns=['Class'])
X_test = X_test.drop(columns=['Class'])

#Question 1
print("-----------------K Nearest Neighbour---------------------\n")
#Using the k nearest neighbour method with k=1,3,5 to classify the data
knn1 = KNeighborsClassifier(n_neighbors=1)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)

#Fit the k-nearest neighbors classifier from the training dataset
knn1.fit(X_train, X_label_train)
knn3.fit(X_train, X_label_train)
knn5.fit(X_train, X_label_train)

#predicting the class of the test data using knn method and storing it
X_label_pred_1 = knn1.predict(X_test)
X_label_pred_3 = knn3.predict(X_test)
X_label_pred_5 = knn5.predict(X_test)

#finding the confusion matrix of each of the predicted dataset where k=1,3,5
cm1 = confusion_matrix(X_label_pred_1, X_label_test)
cm3 = confusion_matrix(X_label_test, X_label_pred_3)
cm5 = confusion_matrix(X_label_test, X_label_pred_5)

#printing the confusion matrix
print("Confusion matrix for K=1:\n",cm1,"\n")
print("Confusion matrix for K=3:\n",cm3,"\n")
print("Confusion matrix for K=5:\n",cm5,"\n")

#printing the classification accuracy of the predicted data
print("Classification accuracy with K=1:", accuracy_score(X_label_test, X_label_pred_1)*100)
print("Classification accuracy with K=3:", accuracy_score(X_label_test, X_label_pred_3)*100)
print("Classification accuracy with K=5:", accuracy_score(X_label_test, X_label_pred_5)*100)

# Question 2
print("\n--------K Nearest Neighbour on normalised data----------\n")
#creating copies of the train and test data for normalisation
X_train_normalised = X_train.copy()
X_test_normalised = X_test.copy()

#normalising the train and test data using minimum and maximum values of the train dataset
for i in X_train.columns:
    minimum = X_train_normalised[i].min()
    maximum = X_train_normalised[i].max()
    diff = maximum-minimum
    X_train_normalised[i] = (X_train_normalised[i] - minimum)/diff
    X_test_normalised[i] = (X_test_normalised[i] - minimum)/diff

#saving the normalised train data and test data as csv files
X_train_normalised['Class'] = X_label_train
X_test_normalised['Class'] = X_label_test
X_train_normalised.to_csv('SteelPlateFaults-train-Normalised.csv',index =False)
X_test_normalised.to_csv('SteelPlateFaults-test-Normalised.csv',index=False)
X_train_normalised = X_train_normalised.drop(columns=['Class'])
X_test_normalised = X_test_normalised.drop(columns=['Class'])

#Using the k nearest neighbour method with k=1,3,5 to classify the normalised data
knn1.fit(X_train_normalised, X_label_train)
knn3.fit(X_train_normalised, X_label_train)
knn5.fit(X_train_normalised, X_label_train)

#predicting the class of the test data using knn method and storing it
X_label_pred_1_normalised = knn1.predict(X_test_normalised)
X_label_pred_3_normalised = knn3.predict(X_test_normalised)
X_label_pred_5_normalised = knn5.predict(X_test_normalised)

#finding the confusion matrix of each of the predicted dataset where k=1,3,5
cm1_normalised = confusion_matrix(X_label_test, X_label_pred_1_normalised)
cm3_normalised = confusion_matrix(X_label_test, X_label_pred_3_normalised)
cm5_normalised = confusion_matrix(X_label_test, X_label_pred_5_normalised)

#printing the confusion matrix
print("Confusion matrix of normalised data for K=1:\n",cm1_normalised,"\n")
print("Confusion matrix of normalised data for K=3:\n",cm3_normalised,"\n")
print("Confusion matrix of normalised data for K=5:\n",cm5_normalised,"\n")

#printing the classification accuracy of the predicted data
print("Classification accuracy on normalised data with K=1:", accuracy_score(X_label_test, X_label_pred_1_normalised)*100)
print("Classification accuracy on normalised data with K=3:", accuracy_score(X_label_test, X_label_pred_3_normalised)*100)
print("Classification accuracy on normalised data with K=5:", accuracy_score(X_label_test, X_label_pred_5_normalised)*100)


#Question 3
print("\n-------------Bayes Classification--------------\n")
#creating a dataframe which contains all the tuples of the train data with class 0
X_train_0 = pd.read_csv('SteelPlateFaults-train.csv')
X_train_0 = X_train_0[X_train_0['Class']==0]
X_train_0 = X_train_0.drop(columns=['Class'])

#creating a dataframe which contains all the tuples of the train data with class 1
X_train_1 = pd.read_csv('SteelPlateFaults-train.csv')
X_train_1 = X_train_1[X_train_1['Class']==1]
X_train_1 = X_train_1.drop(columns=['Class'])

#caluclating the class prior probabilities
prob_0 = X_train_0.shape[0]/(X_train_0.shape[0] + X_train_1.shape[0])   
prob_1 = X_train_1.shape[0]/(X_train_0.shape[0] + X_train_1.shape[0])   

true1=0;false1=0;false0=0;true0=0       #setting the count of true and false outcomes of each class to zero
pred=[]     #list is used to store the predicted data from bayes classification

mean_0 = np.array(X_train_0.mean())
mean_1 = np.array(X_train_1.mean())
cov_0 = X_train_0.cov()
cov_1 = X_train_1.cov()
cov_0_det = np.linalg.det(cov_0)
cov_1_det = np.linalg.det(cov_1)
cov_0_inv = np.linalg.inv(cov_0)
cov_1_inv = np.linalg.inv(cov_1)

#function to estimate the log of the likelihood of a tuple
def log_likelihood(arr,invmat,det):
    exponent = (-0.5 * (np.dot(np.dot(arr.reshape(1,27),invmat),arr))[0])
    p = log((1/det)/((2*pi)**13.5)) + exponent
    return p

for i in range(X_test.shape[0]):
    pred_0 = log(prob_0)        #pred_0 is used to store the log of the likelihood that the class of the new tuple is 0
    pred_1 = log(prob_1)        #pred_1 is used to store the log of the likelihood that the class of the new tuple is 1
    x = np.array(X_test.iloc[i])        #stores the tuple of the test dataframe
    pred_0+=log_likelihood(np.subtract(x,mean_0),cov_0_inv,cov_0_det)
    pred_1+=log_likelihood(np.subtract(x,mean_1),cov_1_inv,cov_1_det)
    if (pred_1>pred_0):         #if pred_1 > pred_0, the predicted value of the tuple is 1
        pred.append(1)
        if (X_label_test[X_label_test.index[i]])==1:        #if the test label is 1, then the predicted value is correct, i.e, a true 1
            true1+=1
        else:                                               #if the test label is 0, then the predicted value is false, i.e, a false 1
            false1+=1
    if (pred_0>pred_1):         #if pred_0 > pred_1, the predicted value of the tuple is 0
        pred.append(0)
        if (X_label_test[X_label_test.index[i]])==0:        #if the test label is 0, then the predicted value is correct, i.e, a true 0
            true0+=1
        else:                                               #if the test label is 1, then the predicted value is false, i.e, a false 0
            false0+=1

confusionmatrix = [[true0,false1],[false0,true1]]
print("Confusion matrix:\n",confusionmatrix[0],"\n",confusionmatrix[1])
accuracy = (true1+true0)/(true1+true0+false1+false0)
print("\nClassification accuracy:",accuracy*100)
print()


#Question 4
print("\n-------------Tabulation of Results---------------\n")
#tabulating the best results from knn, knn on normalised data, and bayes classification
classification_method = ['K Nearest neighbour','K nearest Neighbour on normalised data','Bayes Classification']
predicted_data = [X_label_pred_5,X_label_pred_3_normalised,pred]
accuracy_scores = [accuracy_score(X_label_test, X_label_pred_5)*100,accuracy_score(X_label_test, X_label_pred_3_normalised)*100,accuracy*100]
print(pd.DataFrame.from_dict({'Classification Method':classification_method,'Predicted data':predicted_data,
                              'Accuracy':accuracy_scores}).set_index('Classification Method'))


















