
'''
Scaffolding code for the Machine Learning assignment.
You should complete the provided functions and add more functions and classes as necessary.

You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.
You are welcome to use the pandas library if you know it.
'''
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, mean_squared_error

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    '''

    return [(9776460, 'Rune' 'Leistad'), (10405127, 'Jenny', 'Griffiths')]

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

def build_DecisionTree_classifier(X_training, y_training, max_depth, n_folds):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''

    clf = DecisionTreeClassifier()

    tuned_parameters = [{'max_depth': np.arange(1,max_depth+1)}]

    clf = GridSearchCV(clf,tuned_parameters,cv=n_folds)

    clf.fit(X_training,y_training)

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearestNeighbours_classifier(X_training, y_training, K_max, n_folds):
    '''
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    K: number of neighbors
    @return
	clf : the classifier built in this function
    '''

    clf = KNeighborsClassifier(algorithm='auto')

    tuned_parameters = [{'n_neighbors': np.arange(1,K_max+1)}]

    clf = GridSearchCV(clf,tuned_parameters,cv=n_folds)

    clf.fit(X_training,y_training)

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training, C_min, C_max, n_folds):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.
    @param
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    @return
	clf : the classifier built in this function
    '''

    clf = SVC(gamma='scale')

    tuned_parameters = [{'C': np.logspace(C_min,C_max,num=np.absolute(C_max-C_min))}]

    clf = GridSearchCV(clf,tuned_parameters,cv=n_folds)

    clf.fit(X_training,y_training)

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def generate_model():
    '''
    Generates a NN model based on layers and activations.
    @param
    @return
    model: The model of the Neural Network
    '''
    model = Sequential()

    model.add(Dense(30, input_dim=30))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model

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

    # First finding the best hyperparameter
    clf = KerasClassifier(build_fn=generate_model, verbose=0)
    epochs = [10,50,100] # Amount of iterations over each traning set
    batch_size = [20,40,60,80,100] # Different training sample sizes
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    grid = GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1)

    clf = grid.fit(X_training, y_training) # Training the classifier

    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def evaluate_classifier(clf,clf_name, n_folds, X_train, X_test, y_train, y_test, param):
    '''
    Plot and evaluate results of cross validation for either decision tree, SVM or kNN classifiers
    Evaluate performance of classifier on both train and test data

    @param
    clf: Classifier
    clf_name: String identifier for the classifier
    n_folds: Number folds used in the KFold CV
    X_train: X_train[i,:] is the ith example
    X_test: X_test[i,:] is the ith test example
    y_train: y_train[i] is the class label of X_train[i,:]
    y_test: y_test[i] is the class label of X_test[i,:]
    param: The
    '''

    # Calculate scores
    scores = clf.cv_results_['mean_test_score']
    scores_std = clf.cv_results_['std_test_score']
    std_error = scores_std / np.sqrt(n_folds)

    # Set figure parameters and make figure
    if clf_name == 'kNN':
        x_label = 'K'
        x_axis = np.arange(1,param+1)
        tuned_param = 'n_neighbors'

        plt.figure()
        plt.plot(x_axis, scores + std_error, 'b--o', markersize=3)
        plt.plot(x_axis, scores - std_error, 'b--o', markersize=3)
        plt.plot(x_axis, scores,color='black', marker='o',
             markerfacecolor='blue', markersize=5)

    elif clf_name == 'DT':
        x_label = 'Maximum tree depth'
        x_axis = np.arange(1,param+1)
        tuned_param = 'max_depth'

        plt.figure()
        plt.plot(x_axis, scores + std_error, 'b--o', markersize=3)
        plt.plot(x_axis, scores - std_error, 'b--o', markersize=3)
        plt.plot(x_axis, scores,color='black', marker='o',
             markerfacecolor='blue', markersize=5)

    elif clf_name == 'SVM':
        x_label = 'C'
        param_min = param[0]
        param_max = param[1]
        x_axis = np.logspace(param_min,param_max,num=np.absolute(param_min-param_max))
        tuned_param = 'C'

        plt.figure()
        plt.semilogx(x_axis, scores + std_error, 'b--o', markersize=3)
        plt.semilogx(x_axis, scores - std_error, 'b--o', markersize=3)
        plt.semilogx(x_axis, scores,color='black', marker='o',
             markerfacecolor='blue', markersize=5)

    elif clf_name == 'NNC':
        x_label = 'Epochs'
        x_axis = np.arange(1,param+1)
        tuned_param = 'epochs'

        plt.figure()
        plt.plot(x_axis, scores + std_error, 'b--o', markersize=3)
        plt.plot(x_axis, scores - std_error, 'b--o', markersize=3)
        plt.plot(x_axis, scores,color='black', marker='o',
             markerfacecolor='blue', markersize=5)


    plt.fill_between(x_axis, scores + std_error, scores - std_error, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel('Cross validation score +/- std error')
    plt.title('Cross validation results')

    # Calculate error rates
    pred_train = clf.predict(X_train)
    train_clf_errors = np.sum(y_train!=pred_train)
    train_mse = mean_squared_error(pred_train,y_train)

    pred_test = clf.predict(X_test)
    test_clf_errors = np.sum(y_test!=pred_test)
    test_mse = mean_squared_error(pred_test,y_test)

    # Print summary
    print(clf_name +' \nNumber of errors on training data: ', train_clf_errors, '\nMSE for training data', train_mse)
    print('Number of errors on test data: ', test_clf_errors, '\nMSE for test data', test_mse)
    print('The best choice of ' + x_label + ': ' + str(clf.best_params_[tuned_param]),'\n')

    train_precision = 1 - train_clf_errors/len(y_train)
    test_precision = 1 - test_clf_errors/len(y_test)
    print('Training precision: ', train_precision)
    print('Test precision: ', test_precision)

    # source: https://scikit-learn.org/stable/auto_examples/exercises/plot_cv_diabetes.html#sphx-glr-auto-examples-exercises-plot-cv-diabetes-py

if __name__ == "__main__":
  
    np.random.seed(7)
    X,y = prepare_dataset('medical_records.data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # train and test nearest neighbor classifier
    K_max = 20
    n_folds = 10
    clf = build_NearestNeighbours_classifier(X_train,y_train,K_max,n_folds)
    evaluate_classifier(clf,'kNN',n_folds,X_train,X_test,y_train,y_test,K_max)

    # train and test decision tree classifier
    max_depth = 20
    clf = build_DecisionTree_classifier(X_train, y_train,max_depth,n_folds)
    evaluate_classifier(clf,'DT',n_folds,X_train,X_test,y_train,y_test,max_depth)

    # train and test SVM
    C_min = -4
    C_max = 8
    clf = build_SupportVectorMachine_classifier(X_train, y_train, C_min, C_max,n_folds)
    evaluate_classifier(clf,'SVM',n_folds,X_train,X_test,y_train,y_test,[C_min,C_max])

    # Train and test NN classifier
    nodes = 15
    clf = build_NeuralNetwork_classifier(X_train, y_train)
    evaluate_classifier(clf, 'NNC', n_folds, X_train, X_test, y_train, y_test, nodes)
