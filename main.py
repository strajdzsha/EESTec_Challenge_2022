import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.datasets import load_iris
import pickle

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Train-dataset.csv')

wells = df['WELL'].unique()

well_data = df[df['WELL'] == wells[10]]

MAPPING = {
    'Continental': 1,
    'Transitional': 2,
    'Marine': 3,
}

df['D_Env']=df['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

Feature = df[['X', 'Y', 'MD','GR', 'RT', 'DEN', 'CN','D_Env']]
X = Feature
X= preprocessing.StandardScaler().fit(X).transform(X)
Y = df['LITH_CODE']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 7)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# fig = plt.figure(figsize=(150,150))
# _ = tree.plot_tree(clf, max_depth=3,
#                    filled=True)
# fig.savefig("decision_tree.png")

# predTree = clf.predict(X_test)
# print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

print("Train set accuracy: ", round(metrics.f1_score(y_train, clf.predict(X_train), average = 'micro'), 2))
print("Test set accuracy: ", round(metrics.f1_score(y_test, clf.predict(X_test), average = 'micro'), 2))