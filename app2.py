import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

pd.set_option('display.max_columns', None)

features_df = {
    1: ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class', 'Flight Distance',
        'Seat comfort', 'Departure/Arrival time convenient', 'Food and drink',
        'Gate location', 'Inflight wifi service', 'Inflight entertainment', 'Online support',
        'Ease of Online booking', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
        'Cleanliness', 'Online boarding', 'Departure Delay in Minutes', 'Arrival Delay in Minutes'],
    2: ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
    3: ['center_id', 'city_code', 'region_code', 'op_area']
}
labels = {
    1: 'satisfaction',
    2: 'Drug',
    3: 'center_type'
}
amount_of_clusters = {
    1: 2,
    2: 5,
    3: 3
}


def read_and_prepare_dataset(nr):
    df = None
    if nr == 1:
        df = pd.read_csv('dataset1.csv')
        df = df[:1000]
        for i, data in enumerate(df['satisfaction']):
            if data == "satisfied":
                df.at[i, 'satisfaction'] = 1
            else:
                df.at[i, 'satisfaction'] = 0
        for i, data in enumerate(df['Gender']):
            if data == "Male":
                df.at[i, 'Gender'] = 1
            else:
                df.at[i, 'Gender'] = 0
        for i, data in enumerate(df['Customer Type']):
            if data == "Loyal Customer":
                df.at[i, 'Customer Type'] = 1
            else:
                df.at[i, 'Customer Type'] = 0
        for i, data in enumerate(df['Type of Travel']):
            if data == "Personal Travel":
                df.at[i, 'Type of Travel'] = 1
            else:
                df.at[i, 'Type of Travel'] = 0
        for i, data in enumerate(df['Class']):
            if data == "Eco":
                df.at[i, 'Class'] = 1
            else:
                df.at[i, 'Class'] = 0
        df.dropna(inplace=True)

    elif nr == 2:
        df = pd.read_csv('dataset2.csv')
        for i, data in enumerate(df['Sex']):
            if data == "M":
                df.at[i, 'Sex'] = 1
            else:
                df.at[i, 'Sex'] = 0
        for i, data in enumerate(df['BP']):
            if data == "HIGH":
                df.at[i, 'BP'] = 2
            elif data == "NORMAL":
                df.at[i, 'BP'] = 1
            else:
                df.at[i, 'BP'] = 0
        for i, data in enumerate(df['Cholesterol']):
            if data == "HIGH":
                df.at[i, 'Cholesterol'] = 1
            else:
                df.at[i, 'Cholesterol'] = 0
        for i, data in enumerate(df['Drug']):
            if data == "drugA":
                df.at[i, 'Drug'] = 4
            elif data == "drugB":
                df.at[i, 'Drug'] = 3
            elif data == "drugC":
                df.at[i, 'Drug'] = 2
            elif data == "drugX":
                df.at[i, 'Drug'] = 1
            elif data == "drugY":
                df.at[i, 'Drug'] = 0
        df.dropna(inplace=True)

    elif nr == 3:
        df = pd.read_csv('dataset3.csv')
        for i, data in enumerate(df['center_type']):
            if data == "TYPE_A":
                df.at[i, 'center_type'] = 2
            elif data == "TYPE_B":
                df.at[i, 'center_type'] = 1
            else:
                df.at[i, 'center_type'] = 0
        df.dropna(inplace=True)
    return df


if __name__ == '__main__':
    dataset_nr = 2
    df = read_and_prepare_dataset(dataset_nr)
    x = df.loc[:, features_df[dataset_nr]].values
    y = df.loc[:, labels[dataset_nr]].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents, columns=['f1', 'f2'])

    agg = AgglomerativeClustering(n_clusters=amount_of_clusters[dataset_nr])
    clustering = agg.fit_predict(principalDf)
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(principalDf['f1'], principalDf['f2'], c=clustering, cmap='rainbow')
    fig.suptitle("agglomerative clustering")
    plt.savefig(f"set{dataset_nr}/agg_set{dataset_nr}.png")

    km = KMeans(n_clusters=amount_of_clusters[dataset_nr])
    clustering_k_means = km.fit_predict(principalDf)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("k means")
    plt.scatter(principalDf['f1'], principalDf['f2'], c=clustering_k_means, cmap='rainbow')
    plt.savefig(f"set{dataset_nr}/k_means_set{dataset_nr}.png")

    centroids_km = km.cluster_centers_
    points_km = []
    for i, item in enumerate(principalDf['f1']):
        points_km.append([item, principalDf['f2'].iloc[i]])
    distances_km = cdist(centroids_km, principalDf, 'euclidean')

    percentile = 80
    outliers_km = [[], []]
    for i, item in enumerate(distances_km[0]):
        if (distances_km[0][i] + distances_km[1][i]) / 2 > np.percentile(distances_km, percentile):
            outliers_km[0].append(principalDf['f1'].iloc[i])
            outliers_km[1].append(principalDf['f2'].iloc[i])
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(principalDf['f1'], principalDf['f2'], c=clustering_k_means, cmap='rainbow')
    plt.scatter(outliers_km[0], outliers_km[1], marker='o', s=50, facecolor='#000000')
    fig.suptitle("custom_outliers")
    plt.savefig(f"set{dataset_nr}/finish_km{dataset_nr}.png")

    clf = LocalOutlierFactor(n_neighbors=5)
    clf_predicted = clf.fit_predict(principalDf)
    clf_results =[[],[]]
    for i, item in enumerate(clf.negative_outlier_factor_):
        if item < np.percentile(clf.negative_outlier_factor_, 100-percentile):
            clf_results[0].append(principalDf['f1'].iloc[i])
            clf_results[1].append(principalDf['f2'].iloc[i])
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(principalDf['f1'], principalDf['f2'], c=clustering_k_means, cmap='rainbow')
    plt.scatter(clf_results[0], clf_results[1], marker='o', s=70, facecolor='#000000')
    fig.suptitle("lof_km")
    plt.savefig(f"set{dataset_nr}/lof_km{dataset_nr}.png")

    clf = LocalOutlierFactor(n_neighbors=5)
    clf_predicted = clf.fit_predict(principalDf)
    clf_results = [[], []]
    for i, item in enumerate(clf.negative_outlier_factor_):
        if item < np.percentile(clf.negative_outlier_factor_, 100 - percentile):
            clf_results[0].append(principalDf['f1'].iloc[i])
            clf_results[1].append(principalDf['f2'].iloc[i])
    fig = plt.figure(figsize=(10, 7))
    plt.scatter(principalDf['f1'], principalDf['f2'], c=clustering, cmap='rainbow')
    plt.scatter(clf_results[0], clf_results[1], marker='o', s=70, facecolor='#000000')
    fig.suptitle("lof_agg")
    plt.savefig(f"set{dataset_nr}/lof_agg{dataset_nr}.png")

    eps = 0.5
    clustering_db = DBSCAN(eps=eps, min_samples=2).fit(principalDf)
    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("dbscan")
    plt.scatter(principalDf['f1'], principalDf['f2'], c=clustering_db.labels_, cmap='rainbow')
    plt.savefig(f"set{dataset_nr}/dbscan_set{dataset_nr}.png")
    print("Done!")