from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from scipy.io import arff
import pandas as pd
from pandas.plotting import scatter_matrix
import openCICFlowMeter as openCic
import openGroupCsv
import generateDataset
import os
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from joblib import dump, load
import sys

# CRIAR SCRIPT PARA INSTALAR TODOS OS PRÉ-REQUISITOS COMO PANDAS, KERAS, SCIKIT ETC
# CHAMAR O ARQUIVO QUE ABRE O CIC
# openCic.runCICFlowMeter()

if ((not os.path.isfile('concatened_dataset.csv')) & (not os.path.isfile('validate_dataset.csv'))):
    generateDataset.concatenedToCsv()

concatened_dataset = pd.read_csv('concatened_dataset.csv')
validate_dataset = pd.read_csv('validate_dataset.csv')

print("Gerando datasets para segundo caso de teste")
generateDataset.secondCaseTest()
second_case_test_concatened_dataset = pd.read_csv('second_case_test_concatened_dataset.csv')
second_case_test_validate_dataset = pd.read_csv('second_case_test_validate_dataset.csv')
print("=============================================================================================")

array = concatened_dataset.values
# print(array)
x = array[:,0:23]        # list slicing for attributes. [start:stop:step], def step = 1. in this case [from start:until last instance (,0 until last first col):step = 4 (4 columns to copy and skip last column)]
y = array[:,23]          # list slice for class column

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)

#votingClassifier

decisionTree = DecisionTreeClassifier(max_depth=7)
knn = KNeighborsClassifier()
randomForest = RandomForestClassifier(n_estimators=10)
votingClassifier = VotingClassifier(
    estimators=[
        ('dt', decisionTree),
        ('knn', knn),
        ('rf', randomForest)
    ],
    voting='soft', weights=[1, 2, 1]
)

print('treinando 1')
decisionTree.fit(X_train, Y_train)
print('treinando 2')
knn.fit(X_train, Y_train)
print('treinando 3')
randomForest.fit(X_train, Y_train)
print('voting')
votingClassifier.fit(X_train, Y_train)

# votingModel = dump(votingClassifier, 'second_case_test_voting_classifier.joblib') 
votingClassifier = load('first_case_test_voting_classifier.joblib') 

#fazer a verificação com cada um dos modelos também

print("Validando com DecisionTreeClassifier")
predictions = decisionTree.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(decisionTree, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("Validando com KNeighborsClassifier")
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(knn, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("Validando com RandomForestClassifier")
predictions = randomForest.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(randomForest, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("Validando com voting")
predictions = votingClassifier.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(votingClassifier, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("================== validação do primeiro caso de teste ==================")

predictions = decisionTree.predict(validate_dataset)
print('\nDecisionTreeClassifier')
print(Counter(predictions))

predictions = randomForest.predict(validate_dataset)
print('\nrandomForest')
print(Counter(predictions))

predictions = knn.predict(validate_dataset)
print('\nknn')
print(Counter(predictions))

predictions = votingClassifier.predict(validate_dataset)
print('\nvotingClassifier')
print(Counter(predictions))




###################
array = second_case_test_concatened_dataset.values
# print(array)
x = array[:,0:4]        # list slicing for attributes. [start:stop:step], def step = 1. in this case [from start:until last instance (,0 until last first col):step = 4 (4 columns to copy and skip last column)]
y = array[:,4]          # list slice for class column

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)

#votingClassifier

decisionTree = DecisionTreeClassifier(max_depth=7)
knn = KNeighborsClassifier()
randomForest = RandomForestClassifier(n_estimators=10)
votingClassifier = VotingClassifier(
    estimators=[
        ('dt', decisionTree),
        ('knn', knn),
        ('rf', randomForest)
    ],
    voting='soft', weights=[1, 2, 1]
)

print('treinando 1')
decisionTree.fit(X_train, Y_train)
print('treinando 2')
knn.fit(X_train, Y_train)
print('treinando 3')
randomForest.fit(X_train, Y_train)
print('voting')
votingClassifier.fit(X_train, Y_train)


# votingModel = dump(votingClassifier, 'second_case_test_voting_classifier.joblib') 
votingClassifier = load('second_case_test_voting_classifier.joblib') 
###################

print("Validando com DecisionTreeClassifier")
predictions = decisionTree.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(decisionTree, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("Validando com KNeighborsClassifier")
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(knn, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("Validando com RandomForestClassifier")
predictions = randomForest.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(randomForest, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("Validando com voting")
predictions = votingClassifier.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(votingClassifier, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))

print("================== validação do segundo caso de teste ==================")

predictions = decisionTree.predict(second_case_test_validate_dataset)
print('\nDecisionTreeClassifier')
print(Counter(predictions))

predictions = randomForest.predict(second_case_test_validate_dataset)
print('\nrandomForest')
print(Counter(predictions))

predictions = knn.predict(second_case_test_validate_dataset)
print('\nknn')
print(Counter(predictions))

predictions = votingClassifier.predict(second_case_test_validate_dataset)
print('\nvotingClassifier')
print(Counter(predictions))























# # Compare Algorithms
# https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_probas.html
#votingClassifier


# FAZER VÁRIAS IMAGENS PLT

# validation_size = 0.20  # 20% for the validation set
# seed = 7

# returns tuple of values

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# clf=RandomForestClassifier(n_estimators=20)

# # #Train the model using the training sets y_pred=clf.predict(X_test)
# clf.fit(X_train,y_train)

# y_pred=clf.predict(X_test)

# print("Accuracy:\n", accuracy_score(y_test, y_pred))
# print("Confusion:\n", confusion_matrix(y_test,y_pred))
# print("Report:\n", classification_report(y_test,y_pred))
# print(y_test)
# print(y_pred)

#COMECA AQUI COMO ESTAVA ANTES DE TESTAR O VOTING
# validation_size = 0.25  # 20% for the validation set
# seed = 20
# # returns tuple of values
# X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=validation_size, random_state=seed) # understand what thsis does mroe indepth and write about it. doing some fitting, look at leetcode intro to ML
# # Split arrays or matrices into random train and test subsets. (numpy arr, numpy arr, testsize is the portion of the dataset to use for the test split, random_state is Pseudo-random number generator state used for random sampling.)
# # X_train is for the intances used for training
# # Y_train is for the expected outcome of each instance
# # X_validation is the instances used for validating the model
# # Y_validation is for the expected outcome of each corresponding instance in X_validation

# #5.2 Test harness
# # Test options and evaluation metric
# seed = 7
# scoring = 'accuracy'

# # 5.3 build model
# # Spot Check Algorithms
# models = []
# models.append(('RF', RandomForestClassifier(n_estimators=4)))
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('NB', GaussianNB()))
# # evaluate each model in turn
# results = []
# names = []
# for name, model in models:
# 	kfold = model_selection.KFold(n_splits=2);                                          # sklearn. KFold provides train/test indices to split data in train/test sets. Split dataset into k consecutive folds (>= 2 folds) (without shuffling by default). No shuffle to compare algorithms
# 	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring);       # Evaluate a score by cross validation (model is the algorithm to fit the data,x_train is the data to fit for the model,y_train is target variable to predict; result of the x_train instances,cv is the cross validation splitting strat,scoring is the accuracy of the test set)
# 	results.append(cv_results);                                                                             # store the scores (array) of each run of the cross validation in the result array
# 	names.append(name);                                                                                     # Stores the name of the algorithm for the current result
# 	msg = "%s: %f (%f)" % (name, cv_results.min(), cv_results.std())
# 	print(cv_results);
# 	print(msg)


# # Compare Algorithms
# # fig = plt.figure()
# # fig.suptitle('Algorithm Comparison')
# # ax = fig.add_subplot(111)
# # plt.boxplot(results)
# # ax.set_xticklabels(names)
# # plt.show()

# # 6 Make predictions on validation dataset
# print("Predicting on unseen data with KNN.");
# knn = KNeighborsClassifier();                                   # sklearn lib. stores instances of the training data. Classification is computed from a simple majority vote of the nearest neighbors of each point
# knn.fit(X_train, Y_train);                                      # train the model with the train dataset
# predictions = knn.predict(X_validation);                        # get the predictions using the validation test with this model knn
# print(accuracy_score(Y_validation, predictions));               # compares the validation known answer with the predicted to determine accuracy
# print(confusion_matrix(Y_validation, predictions));             # matrix of accuracy classification where C(0,0) is true negatives, C(1,0) is false negatives, C(1,1) true posivtes, C(0,1) false positvies.
# print(classification_report(Y_validation, predictions));        # text report
# print("X_validation predict ===");
# print(predictions);                                             # array of predicted values

# print("Predicting on unseen data with RF");
# rf = RandomForestClassifier();
# rf.fit(X_train, Y_train);
# predictions = rf.predict(X_validation);
# print(accuracy_score(Y_validation, predictions));
# print(confusion_matrix(Y_validation, predictions));
# print(classification_report(Y_validation, predictions));
# print("X_validation predict ===");
# print(predictions);
#TERMINA AQUI COMO ESTAVA ANTES DE TESTAR O VOTING

# for row_index, (iut, predictions, Y_validation) in enumerate(zip (X_validation, predictions, Y_validation)):
#   if predictions != Y_validation:
#     print('Row', row_index, 'has been classified as ', predictions, 'and should be ', Y_validation)
#     print(X_validation[row_index]);
