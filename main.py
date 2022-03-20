from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from imblearn.over_sampling import RandomOverSampler, SMOTE
import time

start_time = time.time()

import warnings

warnings.filterwarnings("ignore")

lithology_key = {100: 'Clay',
                 200: 'Siltstone/Loess',
                 300: 'Marl',
                 400: 'Clay marl',
                 500: 'Clay sandstone',
                 600: 'Sandstone',
                 700: 'Limestone',
                 800: 'Tight',
                 900: 'Dolomite',
                 1000: 'Coal',
                 1100: 'Coal clay',
                 1200: 'Marly sandstone',
                 1300: 'Sandy marl',
                 1400: 'Marl clay',
                 1500: 'Siltstone clay'
                 }


def despike(yi):
    y = np.copy(yi)

    mean = np.mean(y)
    y[y > 5 * mean] = 5 * mean

    return y


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

    dump(k_means, 'k_means.joblib')

    df['POSITION_CLUSTER'] = np.nan

    for i, well in enumerate(wells):
        df.loc[(df['WELL'] == well), 'POSITION_CLUSTER'] = k_means_labels[i]
        # df[['POSITION_CLUSTER', well]] = k_means_labels[i]

    return df


def show_conf_matrix(y_test, y_pred, classes, test_num):
    # Calculate confusion matrix
    conf = confusion_matrix(y_test, y_pred)

    fig = plt.figure(figsize=(12, 12))
    sns.set(font_scale=1)
    sns.heatmap(conf, annot=True, annot_kws={"size": 16}, fmt="d", linewidths=.5, cmap="YlGnBu", xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted value')
    plt.ylabel('True value')

    plt.savefig(str(test_num) + '.jpg')


df = pd.read_csv('Train-dataset.csv')

encodings = df.groupby('DEPOSITIONAL_ENVIRONMENT')['LITH_CODE'].mean().reset_index()
list_encodings = encodings['LITH_CODE'].values
MAPPING = {
    'Continental': list_encodings[0],
    'Transitional': list_encodings[1],
    'Marine': list_encodings[2],
}

df['D_Env'] = df['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

df = addClusters(df)

encodings = df.groupby('POSITION_CLUSTER')['LITH_CODE'].mean().reset_index()
list_encodings = encodings['LITH_CODE'].values
MAPPING = {
    0: list_encodings[0],
    1: list_encodings[1],
    2: list_encodings[2],
}

df['Pos'] = df['POSITION_CLUSTER'].apply(lambda x: MAPPING[x])

sample_dic = {
    600: 11572,
    400: 8054,
    100: 7716,
    500: 6500,
    1300: 5677,
    1000: 3000,
    300: 2000,
    1200: 2000,
    800: 2000,
    1100: 1500,
    1500: 1500,
    1400: 1000,
    200: 100
}

strategies = ['all', 'minority', 'not majority', 'not minority', sample_dic]

for well_num in [1]:
    random_state = 500
    df_new = df[df['WELL'] != 'Well-' + str(well_num)]
    df_rest = df[df['WELL'] == 'Well-' + str(well_num)]

    feature_list = ['Pos', 'MD', 'GR', 'RT', 'DEN', 'CN', 'D_Env']

    X_new = df_new[feature_list]
    X_rest = df_rest[feature_list]

    Y_new = df_new['LITH_CODE']
    Y_rest = df_rest['LITH_CODE']

    ros = RandomOverSampler(random_state=random_state, sampling_strategy='all')
    X_new, Y_new = ros.fit_resample(X_new, Y_new)

    X_train, X_test, y_train, y_test = train_test_split(X_new, Y_new, test_size=0.2, random_state=random_state)
    scaler = preprocessing.MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_validation = scaler.transform(X_rest)
    y_validation = Y_rest

    target_lithologys = []
    labels = np.sort(y_test.unique())

    for l_code in labels:
        lithology = lithology_key[l_code]
        target_lithologys.append(lithology)

    # the best were lr = 0.2 and lr = 0.5
    # best depth was 10

    lr_rates = [0.2]
    depths = [10]
    n_estimators_list = [50]

    for lr in lr_rates:
        for max_depth in depths:
            clf = GradientBoostingClassifier(n_estimators=50, learning_rate=lr, min_samples_split=300,
                                             min_samples_leaf=50,
                                             max_depth=max_depth, subsample=0.8, random_state=random_state).fit(X_train, y_train)

            print('Finished fitting')

            y_validation_pred = clf.predict(X_validation)

            print("Train set accuracy: ", round(metrics.f1_score(y_train, clf.predict(X_train), average='micro'), 5))
            print("Test set accuracy: ", round(metrics.f1_score(y_test, clf.predict(X_test), average='micro'), 5))
            print("Validation set accuracy: ",
                  round(metrics.f1_score(y_validation, clf.predict(X_validation), average='micro'), 5))
            print("This was well number: " + str(well_num) + " and lr: " + str(lr) + ' and depth: ' + str(max_depth), end='\n')
            #show_conf_matrix(y_validation, y_validation_pred, target_lithologys, test_num=well_num)
            print(time.time() - start_time)

            if well_num == 1:
                dump(clf, 'model_new_1.joblib')
                dump(scaler, 'scaler_new_1.joblib')
            print('FINISHED')
