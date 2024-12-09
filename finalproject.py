import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

dataframe = pd.read_csv('gym_dataset.csv')
# features = ["Age","Weight (kg)","Height (m)","BMI","Workout_Frequency (days/week)"]
features = dataframe.columns[~dataframe.columns.isin(['Gender', 'Workout_Type'])]
outlier_cols = ["Weight (kg)","BMI","Calories_Burned"]
sb = st.sidebar
sections = ["Overview","Data Exploration","Data Preparation","Analysis and Insights","Conclusions and Recommendations Section"]

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

with sb:
    st.title("Data Science Final Project")
    st.markdown('<h5 style="color:gray;">Jandel Macabecha</h5>', unsafe_allow_html=True)
    st.divider()
    var_radio = st.radio("Sections",sections)


if var_radio == sections[0]:
    st.subheader("Overview")
    st.write("This dataset is called Gym Members Exercise Tracking Dataset taken from Kaggle. The dataset can be used to analyze gym members workout habits and performance metrics based on features like Age, Weight, Height, BMI and Workout Frequency. The technique used in this project is Clustering.")
    st.write(dataframe.head())
    st.write("Features within the dataset.")
    st.write(dataframe.columns)
    st.write(f'Null values found in the dataset')
    st.write(dataframe.isnull().sum())
elif var_radio == sections[1]:
    dataCopy = dataframe.copy()
    dataCopy = dataCopy.drop(columns=["Gender", "Workout_Type"])
    st.write(dataCopy.describe())
    left_col, right_col = st.columns(2)
    for column in dataCopy.columns:
        with left_col:
            fig, ax = plt.subplots(figsize=(5,5))
            sns.boxplot(data=dataCopy[column], ax=ax)
            st.subheader(f"{column} boxplot")
            st.pyplot(fig)
        with right_col:
            fig, ax = plt.subplots(figsize=(5, 5))
            if column == features[-1]:
                sns.histplot(dataCopy[column], bins=5, kde=True, ax=ax)
            else:
                sns.histplot(dataCopy[column], bins=10, kde=True, ax=ax)
            ax.set_title(f"{column} Histogram")
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
    st.divider()
    correlation_matrix = dataCopy.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

    # st.divider()
    # pairplot_fig = sns.pairplot(dataCopy)
    # st.pyplot(pairplot_fig)
elif var_radio == sections[2]:
    st.subheader("Chosen correlated features.")
    dataCopy = dataframe.copy()
    dataCopy = dataCopy[features]
    correlation_matrix = dataCopy.corr()
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    st.pyplot(fig)

    for column in outlier_cols:
        outliers = detect_outliers_iqr(dataCopy[column])
        dataCopy.loc[outliers, column] = np.nan
    st.subheader("Detected Outliers")
    st.write(dataCopy.isna().sum())
    st.subheader("After removing outliers")
    dataCopy = dataCopy.dropna()
    st.write(dataCopy.isna().sum())
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(5,5))
        sns.boxplot(data=dataCopy[outlier_cols[0]],ax=ax1)
        st.write(f'${outlier_cols[0]} boxplot')
        st.pyplot(fig1)
    with col2:
        fig2, ax1 = plt.subplots(figsize=(5,5))
        sns.boxplot(data=dataCopy[outlier_cols[1]],ax=ax1)
        st.write(f'${outlier_cols[1]} boxplot')
        st.pyplot(fig2)
    with col3:
        fig3, ax1 = plt.subplots(figsize=(5,5))
        sns.boxplot(data=dataCopy[outlier_cols[2]],ax=ax1)
        st.write(f'${outlier_cols[2]} boxplot')
        st.pyplot(fig3)
    st.divider()
    dimension_flat = PCA(n_components=2).fit_transform(dataCopy)
    figPCA, axPCA = plt.subplots(figsize=(10,8))
    sns.scatterplot(x=dimension_flat[:,0], y=dimension_flat[:,1])
    plt.title("Dimension Reduced via PCA")
    st.pyplot(figPCA)
elif var_radio == sections[3]:
    dataCopy = dataframe.copy()
    dataCopy = dataCopy[features]

    for column in outlier_cols:
        outliers = detect_outliers_iqr(dataCopy[column])
        dataCopy.loc[outliers, column] = np.nan
    dataCopy = dataCopy.dropna()
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    dimension_flat = pca.fit_transform(dataCopy)
    df_pca = pd.DataFrame(data=dimension_flat, columns=['PC1','PC2'])
    inertia = []
    for i in range(10):
        kmeans = KMeans(n_clusters=i+1,random_state=42)
        df_pca['Cluster'] = kmeans.fit_predict(dataCopy)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        centers_pca = pca.transform(centers)

        df_centers = pd.DataFrame(data=centers_pca,columns=['PC1','PC2'])
        df_centers['Cluster'] = range(len(df_centers))
        st.write("Cluster Centers in PCA space:")
        st.write(df_centers)
        inertia.append(kmeans.inertia_)
        fig, ax = plt.subplots(figsize=(10, 6))
        # sns.scatterplot(x=dimension_flat[:,0],y=dimension_flat[:,1],palette="deep",hue=labels,alpha=0.7)
        # plt.title(f'KMeans Clustering with n_clusters={i+1}')
        plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', alpha=0.5)
        plt.scatter(df_centers['PC1'], df_centers['PC2'], c='red', marker='X', s=200, label='Centroids')
        plt.title('PCA of Iris Dataset with K-Means Clustering')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        st.pyplot(fig)
        st.write(f'Finished after {kmeans.n_iter_} iterations')

        df_original = pd.DataFrame(data=dataCopy, columns=features)
        df_original['Cluster'] = kmeans.labels_

        cluster_features = df_original.groupby('Cluster').mean()
        st.write("Mean features for each cluster:")
        st.write(cluster_features)
        st.divider()

    figElb, axElb = plt.subplots(figsize=(10,6))
    plt.plot(range(1,11),inertia,marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Sum of Squared Distances)')
    plt.xticks(range(1,11))
    plt.grid()
    st.pyplot(figElb)
elif var_radio == sections[4]:
    st.subheader("As shown on the Elbow Method for Optimal k, I chose a value of 5 for k or number of clusters.")
    dataCopy = dataframe.copy()
    dataCopy = dataCopy[features]

    for column in outlier_cols:
        outliers = detect_outliers_iqr(dataCopy[column])
        dataCopy.loc[outliers, column] = np.nan
    dataCopy = dataCopy.dropna()
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    dimension_flat = pca.fit_transform(dataCopy)
    df_pca = pd.DataFrame(data=dimension_flat, columns=['PC1','PC2'])
    kmeans = KMeans(n_clusters=5,random_state=42)
    df_pca['Cluster'] = kmeans.fit_predict(dataCopy)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    centers_pca = pca.transform(centers)

    df_centers = pd.DataFrame(data=centers_pca,columns=['PC1','PC2'])
    df_centers['Cluster'] = range(len(df_centers))
    fig, ax = plt.subplots(figsize=(10, 6))
    # sns.scatterplot(x=dimension_flat[:,0],y=dimension_flat[:,1],palette="deep",hue=labels,alpha=0.7)
    # plt.title(f'KMeans Clustering with n_clusters={i+1}')
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'], cmap='viridis', alpha=0.5)
    plt.scatter(df_centers['PC1'], df_centers['PC2'], c='red', marker='X', s=200, label='Centroids')
    plt.title('PCA of Iris Dataset with K-Means Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    st.pyplot(fig)

    df_original = pd.DataFrame(data=dataCopy, columns=features)
    df_original['Cluster'] = kmeans.labels_

    cluster_features = df_original.groupby('Cluster').mean()
    st.write("Mean features for each cluster:")
    st.write(cluster_features)

    # Heatmap of mean features by cluster
    plt.figure(figsize=(10, 6))
    sns.heatmap(cluster_features, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Mean Features by Cluster')
    plt.xlabel('Features')
    plt.ylabel('Cluster')
    st.pyplot(plt)
    st.divider()
    st.subheader("Conclusions")
    st.write("As shown in this heatmap for mean features by cluster, An approximation via mean is shown as values for each feature.")
    st.write("We could say that for cluster 0, people's ages are approximately 40, they weigh approximately 70 kilograms, and stands approximately 1.71 meters and so on and so forth for the other feature values.")
    st.write("The data can be used to group gym members to tailor interventions based on the characteristics of each cluster. For example, Cluster 1 might benefit from weight management programs, while Cluster 0 might focus on maintaining their healthy lifestyle.")
    st.subheader("Recommendations")
    st.write("These insights can be explored further by investigating the health implications of the differences in BMI and weight")
    st.write("Consider other lifestyle factors that could influence these clusters, such as diet, sleep, or stress levels.")
