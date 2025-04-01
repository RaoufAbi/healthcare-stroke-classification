from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("D:\1MSTER IATI\exam\iris.csv")



dataset_path = r"D:\1MSTER IATI\exam\iris.csv"
data = pd.read_csv(dataset_path)

# Supprimer les lignes contenant des valeurs NaN
data.dropna(inplace=True)

# Diviser les données en features et target
X = data.drop(columns=['id', 'class'])
y = data['stroke']
# Gestion des données déséquilibrées


gnb = GaussianNB()
gnb.fit(X, y)

# Prédiction avec validation croisée
y_pred = cross_val_predict(gnb, X, y, cv=10)

conf_matrix = confusion_matrix(y, y_pred)

# Calcul des mesures de performance à partir de la matrice de confusion
TP = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TN = conf_matrix[1, 1]

# Calcul de la précision
precision = TP / (TP + FP)

# Calcul du rappel
recall = TP / (TP + FN)

# Calcul du F1-score
f1_score = 2 * (precision * recall) / (precision + recall)

# Calcul de l'exactitude
accuracy = (TN + TP) / (TN + FP + FN + TP)

# Affichage des mesures de performance
print("* BN RESULTS (Cross Validation) :")
print("Accuracy:", accuracy)
print("Précision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Stroke', 'No stroke'], yticklabels=['Stroke', 'No stroke'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
