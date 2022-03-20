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


clf = load('model_new.joblib')
scaler = load('scaler_new.joblib')
test_df = pd.read_csv('Test-dataset.csv')

MAPPING = {
    'Continental': 500,
    'Transitional': 600,
    'Marine': 677,
}

test_df['D_Env'] = test_df['DEPOSITIONAL_ENVIRONMENT'].apply(lambda x: MAPPING[x])

test_df = addClusters(test_df)

MAPPING = {
    0: 630,
    1: 565,
    2: 538,
}
test_df['Pos'] = test_df['POSITION_CLUSTER'].apply(lambda x: MAPPING[x])

feature_list = ['Pos', 'MD', 'GR', 'RT', 'DEN', 'CN', 'D_Env']

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