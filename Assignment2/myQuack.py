
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
#    return [ (1234567, 'Ada', 'Lovelace'), (1234568, 'Grace', 'Hopper'), (1234569, 'Eva', 'Tardos') ]
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

def build_DecisionTree_classifier(X_training, y_training):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearestNeighbours_classifier(X_training, y_training,K):
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
    clf = KNeighborsClassifier(n_neighbors=K,algorithm='auto').fit(X_training,y_training)
    
    
    return clf

def test_kNN(K_values, X_train, X_test, y_train, y_test):
    """
    @param
    K_values: Nearest neighbor values to be tested
    """
    error = []
    min_error = math.inf
    
    for K in K_values:  
        clf = build_NearestNeighbours_classifier(X_train, y_train, K)
        pred_i = clf.predict(X_test)
        mean_error = np.mean(pred_i != y_test)
        if mean_error < min_error:
            min_error = mean_error
            best_k = K
        error.append(mean_error)
        #print(classification_report(y_test, y_pred)) 
    
    plt.figure(figsize=(12, 6))  
    plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')  
    plt.xlabel('K Value')  
    plt.ylabel('Mean Error')
    
    return best_k


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
    K_values = np.arange(1,40)
    best_k = test_kNN(K_values, X_train, X_test, y_train, y_test)
    print('One possible optimal value of K is: ' + str(best_k))
    
    
    
    
    #for precision evaluation of algorithm
    # https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/
    #print(confusion_matrix(y_test, y_pred))  
    #print(classification_report(y_test, y_pred)) 
    


