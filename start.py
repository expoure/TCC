import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor
from scipy.io import arff
import pandas as pd
from pandas.plotting import scatter_matrix
import openCICFlowMeter as openCic
import openGroupCsv
import generateDataset
import os
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


# CRIAR SCRIPT PARA INSTALAR TODOS OS PRÉ-REQUISITOS COMO PANDAS, KERAS, SCIKIT ETC
# CHAMAR O ARQUIVO QUE ABRE O CIC
# openCic.runCICFlowMeter()
# teste_csv = pd.read_csv('csv_result-TimeBasedFeatures-Dataset-120s-AllinOne.csv')
# APÓS FECHAR, ABRIR O ARQUIVO QUE FOI SALVO NA PASTA 'DATA': dataset = pd.read_csv('data/daily/*.csv')
# ESSE ARQUIVO SERÁ UTILIZADO NO MÉTODO... PORÉM
# DURANTE O TREINAMENTO PRECISO VER QUAIS COLUNAS VOU UTILIZAR, SENDO ASSIM
# TALVEZ SE TORNE NECESSÁRIO FAZER UM TRATAMENTO DESSE CSV, EXCLUINDO AS COLUNAS
# DESNECESSÁRIAS, MUDANDO O LABEL ETC

# if os.path.isfile('concatened_dataset.csv'):
#     concatened_dataset = pd.read_csv('concatened_dataset.csv')
# else:
generateDataset.concatenedToCsv()
concatened_dataset = pd.read_csv('concatened_dataset.csv')


print("=============================================================================================")


array = concatened_dataset.values
# print(array)
x = array[:,0:22]        # list slicing for attributes. [start:stop:step], def step = 1. in this case [from start:until last instance (,0 until last first col):step = 4 (4 columns to copy and skip last column)]
y = array[:,23]          # list slice for class column

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(x, y, test_size=0.25, random_state=0)

#votingClassifier

decisionTree = DecisionTreeClassifier(max_depth=4)
knn = KNeighborsClassifier(n_neighbors=7)
randomForest = RandomForestClassifier(n_estimators=4)
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

#fazer a verificação com cada um dos modelos também

print("Validando com DecisionTreeClassifier")
predictions = decisionTree.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(decisionTree, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))
print(predictions)

print("Validando com KNeighborsClassifier")
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(knn, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))
print(predictions)

print("Validando com RandomForestClassifier")
predictions = randomForest.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
# plot_confusion_matrix(randomForest, X_validation, Y_validation)
# plt.show()
print(classification_report(Y_validation, predictions))
print(predictions)

print("Validando com voting")
predictions = votingClassifier.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
plot_confusion_matrix(votingClassifier, X_validation, Y_validation)
plt.show()
print(classification_report(Y_validation, predictions))
print(predictions)


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

# for row_index, (input, predictions, Y_validation) in enumerate(zip (X_validation, predictions, Y_validation)):
#   if predictions != Y_validation:
#     print('Row', row_index, 'has been classified as ', predictions, 'and should be ', Y_validation)
#     print(X_validation[row_index]);

# pop = pd.read_csv('data/daily/2020-10-09_Flow.csv')
# pop = pop[
#     [
#         'Flow Duration',
#         'Fwd IAT Total',
#         'Bwd IAT Total',
#         'Fwd IAT Min',
#         'Bwd IAT Min',
#         'Fwd IAT Max',
#         'Bwd IAT Max',
#         'Fwd IAT Mean',
#         'Bwd IAT Mean',
#         'Flow Packets/s',
#         'Flow Bytes/s',
#         'Flow IAT Min',
#         'Flow IAT Max',
#         'Flow IAT Mean',
#         'Flow IAT Std',
#         'Active Min',
#         'Active Mean',
#         'Active Max',
#         'Active Std',
#         'Idle Min',
#         'Idle Mean',
#         'Idle Max',
#         'Idle Std',
#     ]
# ].copy()

# pop.to_csv(r'pop.csv', index = False, decimal='.')

# predictions_test = clf.predict(pop)
# index = 2
# for result in predictions_test:
#     print("{} : {}".format(index, result))
#     index += 1


# print(test.mean())
# test.to_csv(r'test.csv', index = False, decimal='.')