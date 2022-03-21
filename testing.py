from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn import preprocessing
from joblib import dump, load
from sklearn import metrics


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


def addClusters(df):
    k_means = load('k_means.joblib')
    wells = df['WELL'].unique()
    X = []

    for i, well in enumerate(wells):
        well_df = (df.loc[df['WELL'] == well])
        X.append([well_df['X'].values[0], well_df['Y'].values[0]])
    X = np.asarray(X)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_tran = preprocessing.StandardScaler().fit_transform(X)

    df['POSITION_CLUSTER'] = np.nan

    k_means_labels = k_means.predict(X_tran)

    for i, well in enumerate(wells):
        df.loc[(df['WELL'] == well), 'POSITION_CLUSTER'] = k_means_labels[i]
        # df[['POSITION_CLUSTER', well]] = k_means_labels[i]

    return df


def AddClusterRock(df):
    k_means_rock = load('k_means_rock.joblib')

    Feature = df[['MD', 'GR', 'DEN', 'CN']]
    X = Feature.values

    scaler = preprocessing.MaxAbsScaler()
    X = scaler.fit_transform(X)

    k_means_labels = k_means_rock.predict(X)
    labels = pd.DataFrame(k_means_labels)

    clusterNum = 15
    df['Cluster'] = labels

    MAPPING = {0: 772.0997123681688, 1: 402.3287300297861, 2: 649.7727272727273, 3: 731.9670631290028, 4: 602.8231797919763, 5: 468.848167539267, 6: 550.2277432712216, 7: 393.0073126142596, 8: 680.8269152817187, 9: 662.7157652474108, 10: 298.77675840978594, 11: 597.19563765858, 12: 686.3690577611177, 13: 742.1259842519685, 14: 641.016548463357}
    df['Cluster'] = df['Cluster'].apply(lambda x: MAPPING[x])

    return df


clf = load('model_cluster.joblib') # ovaj je 0.87
scaler = load('scaler_cluster.joblib')
test_df = pd.read_csv('Test-dataset.csv')

MAPPING = {
    'Continental': 500,
    'Transitional': 600,
    'Marine': 677,
}

test_df['D_Env'] = test_df['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

test_df = addClusters(test_df)
test_df = AddClusterRock(test_df)

MAPPING = {
    0: 630,
    1: 565,
    2: 538,
}
test_df['Pos'] = test_df['POSITION_CLUSTER'].apply(lambda x: MAPPING[x])

print(test_df.head(10))

feature_list = ['Pos', 'MD', 'GR', 'RT', 'DEN', 'CN', 'D_Env', 'Cluster']

X = test_df[feature_list]
X = scaler.transform(X)

y_hat = clf.predict(X)

results = pd.DataFrame()
results['Id'] = np.arange(1, len(y_hat)+1)
results['LITH_CODE'] = y_hat
print(results.head())
results.to_csv('submission.csv', index=False)

# old_df = pd.read_csv('old_submission.csv')
# y_hat_old = old_df['LITH_CODE']
#
# print("Train set accuracy: ", round(metrics.f1_score(y_hat, y_hat_old, average='micro'), 5))