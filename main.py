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
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings("ignore")


def show_conf_matrix(y_test, y_pred, classes):
    # Calculate confusion matrix
    conf = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 12))
    sns.set(font_scale=1)
    sns.heatmap(conf, annot=True, annot_kws={"size": 16}, fmt="d", linewidths=.5, cmap="YlGnBu", xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted value')
    plt.ylabel('True value')

    plt.show()


def plotDecisionTree():
    fig = plt.figure(figsize=(100, 100))
    _ = tree.plot_tree(clf, max_depth=3, feature_names=['X', 'Y', 'MD', 'GR', 'RT', 'DEN', 'CN', 'D_Env'],
                       filled=True)
    fig.savefig("decision_tree.png")


df = pd.read_csv('Train-dataset.csv')

wells = df['WELL'].unique()

well_data = df[df['WELL'] == wells[10]]

MAPPING = {
    'Continental': 1,
    'Transitional': 2,
    'Marine': 3,
}

df['D_Env'] = df['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

Feature = df[['X', 'Y', 'MD', 'GR', 'RT', 'DEN', 'CN', 'D_Env']]
X = Feature
X= preprocessing.StandardScaler().fit(X).transform(X)
Y = df['LITH_CODE']
print(Y.value_counts())

oversample = RandomOverSampler(sampling_strategy='not majority')
X, Y = oversample.fit_resample(X, Y)
print(Y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=15)
clf = clf.fit(X_train, y_train)


print("Train set accuracy: ", round(metrics.f1_score(y_train, clf.predict(X_train), average = 'micro'), 5))
print("Test set accuracy: ", round(metrics.f1_score(y_test, clf.predict(X_test), average = 'micro'), 5))
# PLOTTING CONFUSION MATRIX
# target_lithologys = []
# labels = np.sort(y_test.unique())
#
# lithology_key = {100: 'Clay',
#                  200: 'Siltstone/Loess',
#                  300: 'Marl',
#                  400: 'Clay marl',
#                  500: 'Clay sandstone',
#                  600: 'Sandstone',
#                  700: 'Limestone',
#                  800: 'Tight',
#                  900: 'Dolomite',
#                  1000: 'Coal',
#                  1100: 'Coal clay',
#                  1200: 'Marly sandstone',
#                  1300: 'Sandy marl',
#                  1400: 'Marl clay',
#                  1500: 'Siltstone clay'
#                   }
#
# for l_code in labels:
#     lithology = lithology_key[l_code]
#     target_lithologys.append(lithology)
#
# show_conf_matrix(y_predict, y_test, target_lithologys)
