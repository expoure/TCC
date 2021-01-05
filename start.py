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

# CHAMAR O ARQUIVO QUE ABRE O CIC
# openCic.runCICFlowMeter()

if ((not os.path.isfile('concatened_dataset.csv')) & (not os.path.isfile('validate_dataset.csv'))):
    generateDataset.concatenedToCsv()

concatened_dataset = pd.read_csv('concatened_dataset.csv')
validate_dataset = pd.read_csv('validate_dataset.csv')

second_case_test_concatened_dataset = pd.read_csv('second_case_test_concatened_dataset.csv')
second_case_test_validate_dataset = pd.read_csv('second_case_test_validate_dataset.csv')
print("=============================================================================================")

array = concatened_dataset.values
# print(array)
x = array[:,0:23]
y = array[:,23]

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
x = array[:,0:4]
y = array[:,4]

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

