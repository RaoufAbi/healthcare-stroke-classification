from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("healthcare-dataset-stroke-data.csv")

data['smoking_status'] = data['smoking_status'].replace('Unknown', np.nan)

data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})
data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})
data['smoking_status'] = data['smoking_status'].map({'smokes': 1, 'never smoked': 0, 'formerly smoked': 2, 'Unknown': 3})
data['work_type'] = data['work_type'].map({'self-employed': 1, 'govt_job': 2, 'Private': 3 ,'children': 4,'Never_worked': 5})
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})

data.dropna(inplace=True)

X = data[['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Appliquer l'algorithme K-means clustering sur les données réduites en dimensionnalité
kmeans_pca = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_pca.fit(X_pca)
labels_pca = kmeans_pca.labels_

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_pca, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering after PCA')
plt.show()
