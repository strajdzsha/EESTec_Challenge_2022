from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import cross_val_score

import warnings


def addClusters(df):
    wells = df['WELL'].unique()
    X = []

    for i, well in enumerate(wells):
        well_df = (df.loc[df['WELL'] == well])
        X.append([well_df['X'].values[0], well_df['Y'].values[0]])
    X = np.asarray(X)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_tran = preprocessing.StandardScaler().fit_transform(X)

    clusterNum = 3
    k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12)
    k_means.fit(X_tran)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_

    df['POSITION_CLUSTER'] = np.nan

    for i, well in enumerate(wells):
        df.loc[(df['WELL'] == well), 'POSITION_CLUSTER'] = k_means_labels[i]
        # df[['POSITION_CLUSTER', well]] = k_means_labels[i]

    return df


warnings.filterwarnings("ignore")

df = pd.read_csv('Train-dataset.csv')


encodings = df.groupby('DEPOSITIONAL_ENVIRONMENT')['LITH_CODE'].mean().reset_index()
list_encodings = encodings['LITH_CODE'].values
MAPPING = {
    'Continental': list_encodings[0],
    'Transitional': list_encodings[1],
    'Marine': list_encodings[2],
}

#ata = df.merge(encodings, how='left', on='DEPOSITIONAL_ENVIRONMENT')
#df.drop('DEPOSITIONAL_ENVIRONMENT', axis=1, inplace=True)

df['D_Env'] = df['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

df = addClusters(df)


fig, ax = plt.subplots(figsize=(7, 20))
sns.heatmap(df.corr().iloc[-1,:].values.reshape(-1, 1).round(2), annot=True, cmap='RdBu')
plt.title("Feature Correlation", fontsize=15)
# ax.set_yticklabels(df.columns.tolist(), rotation=0)
# ax.set_xticklabels(['price_range'])
plt.show()
for well_num in range(1,2):
    df_new = df[df['WELL'] != 'Well-' + str(well_num)]
    df_rest = df[df['WELL'] == 'Well-' + str(well_num)]

    feature_list = ['POSITION_CLUSTER', 'MD', 'GR', 'RT', 'DEN', 'CN', 'D_Env']

    X_new = df_new[feature_list]
    X_rest = df_rest[feature_list]

    Y_new = df_new['LITH_CODE']
    Y_rest = df_rest['LITH_CODE']

    X_train, X_test, y_train, y_test = train_test_split(X_new, Y_new, test_size=0.01)
    scaler = preprocessing.MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_validation = scaler.transform(X_rest)
    y_validation = Y_rest

    # lr_rates = [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
    # for lr in lr_rates:
    #     clf = GradientBoostingClassifier(n_estimators=20, learning_rate=lr,
    #                                      max_depth=5).fit(X_train, y_train)
    #
    #     print("Train set accuracy: ", round(metrics.f1_score(y_train, clf.predict(X_train), average = 'micro'), 5))
    #     print("Test set accuracy: ", round(metrics.f1_score(y_test, clf.predict(X_test), average = 'micro'), 5))

    # the best were lr = 0.2 and lr = 0.5


    clf = GradientBoostingClassifier(n_estimators=50, learning_rate=0.2,
                                     max_depth=5).fit(X_train, y_train)

    print('Finished fitting')

    print("Train set accuracy: ", round(metrics.f1_score(y_train, clf.predict(X_train), average='micro'), 5))
    print("Test set accuracy: ", round(metrics.f1_score(y_test, clf.predict(X_test), average='micro'), 5))
    print("Validation set accuracy: ", round(metrics.f1_score(y_validation, clf.predict(X_validation), average='micro'), 5))
    print("This was well number: " + str(well_num), end='\n')
