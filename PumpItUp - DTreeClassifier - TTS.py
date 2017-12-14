"""
Using Decision Tree Classifier - classify the pump status
Uses multifeatures
Use decision tree as the sample size is larger, will yield better results than SVM.
Extra - do train-test split because 3 submissions a day only
"""

import matplotlib
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import train_test_split

featuresdf = pd.read_csv("TrainingSetValues.csv")
labelsdf = pd.read_csv("TrainingSetLabels.csv")
testfeaturesdf = pd.read_csv("TestSetValues.csv")
submission = pd.read_csv("SubmissionFormat.csv")

featuresdf['status_group'] = labelsdf['status_group'].values
yvals = []
for i in featuresdf['status_group'].unique():
    s = i.replace(' ', '_')
    yvals.append(s)
    featuresdf[s] = pd.get_dummies(featuresdf['status_group'])[i]

f = featuresdf['status_group'].unique()
def relabel(row):
    for i in range(len(f)):
        if row['status_group'] == f[i]:
            return i
featuresdf['status_group_relabel'] = featuresdf.apply(lambda row: relabel(row), axis = 1)

X = featuresdf[['amount_tsh', 'population', 'num_private', 'construction_year']]
# Xre = X.reshape(-1,1)
y = np.array(featuresdf['status_group_relabel'])
yre = y.reshape(-1,1)
x = testfeaturesdf[['amount_tsh', 'population', 'num_private', 'construction_year']]

X_train, X_test, y_train, y_test = train_test_split(X, yre, test_size = 0.33,random_state = 42)
# output = OneVsRestClassifier(LinearSVC(random_state=0)).fit(Xre, yre).predict(Xre)
clf = tree.DecisionTreeClassifier(min_samples_leaf=100).fit(X_train, y_train)
output2 = clf.predict(X_test)

good = 0
for i in range(len(output2)):
    if output2[i] == y_test[i]:
        good += 1
        
accuracy = good/len(output2)

relabel = []
for i in output2:
    if i == 0:
        relabel.append("functional")
    elif i == 1:
        relabel.append("non functional")
    elif i == 2:
        relabel.append("functional needs repair")
        
        
print (accuracy)
# submission['status_group'] = relabel
# print (relabel.head())
# submission.to_csv('test-1.csv', sep = ',', index = False)