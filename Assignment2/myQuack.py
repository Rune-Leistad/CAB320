
'''

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.


'''
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    '''
    
    return [(9776460, 'Rune' 'Leistad'), (12345678, 'Jenny', 'Griffiths')]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    
    records = open(dataset_path,'r')
    y = []
    X = []
    
    for line in records.readlines():
        record = line.rstrip().split(',')
        if record[1]=='M':
            y.append(1)
        else:
            y.append(0)
        
        X.append(list(record[2:]))
    
    X = np.array(X)
    X = X.astype(np.float64)
    
    y = np.array(y)
    y = y.astype(np.float64)
    
    return X, y

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier(X_training, y_training, max_depth):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"  
    clf = DecisionTreeClassifier()
    
    tuned_parameters = [{'max_depth': np.arange(1,max_depth+1)}]
    
    clf = GridSearchCV(clf,tuned_parameters)
    
    clf.fit(X_training,y_training)
    
    return clf

def DT_evaluation(clf,x_axis,n_folds,X_train,X_test,y_train,y_test):
    
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']
    std_error = scores_std / np.sqrt(n_folds)
    
    plt.figure()
    plt.plot(x_axis, scores + std_error, 'b--o', markersize=3)
    plt.plot(x_axis, scores - std_error, 'b--o', markersize=3)
    plt.plot(x_axis, scores,color='black', marker='o',  
             markerfacecolor='blue', markersize=5)
    plt.fill_between(x_axis, scores + std_error, scores - std_error, alpha=0.2)
    plt.xlabel('Maximum tree depth')
    plt.ylabel('Cross validation score +/- std error')
    plt.title('Cross validation results')

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    
    print('Classification report for training data: \n', classification_report(y_train, pred_train))
    print('Classification report for test data: \n',  classification_report(y_test, pred_test))
    print('The best choice of depth: ' + str(clf.best_params_['max_depth']))
    # source: https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py

    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearestNeighbours_classifier(X_training, y_training,K_max):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    K: number of neighbors

    @return
	clf : the classifier built in this function
    '''
    
    ##         "INSERT YOUR CODE HERE"   
    clf = KNeighborsClassifier(algorithm='auto')
    
    tuned_parameters = [{'n_neighbors': np.arange(1,K_max+1)}]
    
    clf = GridSearchCV(clf,tuned_parameters,cv=5)
    
    clf.fit(X_training,y_training)
    
    return clf

def kNN_evaluation(clf,x_axis,n_folds,X_train,X_test,y_train,y_test):
    '''
    Plot and evaluate results of cross validation
    Evaluate performance of classifier on both train and test data
    
    @param
    clf: Classifier
    x_axis: Vector of K-values for ploting against CV scores
    n_folds: Number folds used in the KFold CV
    X_train: X_train[i,:] is the ith example
    X_test: X_test[i,:] is the ith test example
    y_train: y_train[i] is the class label of X_train[i,:]
    y_test: y_test[i] is the class label of X_test[i,:]
    '''
    
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']
    std_error = scores_std / np.sqrt(n_folds)
    
    plt.figure()
    plt.plot(x_axis, scores + std_error, 'b--o', markersize=3)
    plt.plot(x_axis, scores - std_error, 'b--o', markersize=3)
    plt.plot(x_axis, scores,color='black', marker='o',  
             markerfacecolor='blue', markersize=5)
    plt.fill_between(x_axis, scores + std_error, scores - std_error, alpha=0.2)
    plt.xlabel('K')
    plt.ylabel('Cross validation score +/- std error')
    plt.title('Cross validation results')
    
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    
    print('Classification report for training data: \n', classification_report(y_train, pred_train))
    print('Classification report for test data: \n',  classification_report(y_test, pred_test))
    print('The best choice of K: ' + str(clf.best_params_['n_neighbors']))
    # source: https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    
    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network with two dense hidden layers classifier 
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.


    # call your functions here
    # prepare data sets
    X,y = prepare_dataset('medical_records.data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    
    # train and test nearest neighbor classifier
    K_max = 20 
    clf = build_NearestNeighbours_classifier(X_train,y_train,K_max)
    kNN_evaluation(clf,np.arange(1,K_max+1),5,X_train,X_test,y_train,y_test)
    
    # train and test decision tree classifier
    max_depth = 20
    clf = build_DecisionTree_classifier(X_train, y_train,max_depth)
    DT_evaluation(clf,np.arange(1,max_depth+1),n_folds,X_train,X_test,y_train,y_test)

    
    
