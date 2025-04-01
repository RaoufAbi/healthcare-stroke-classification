from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("healthcare-dataset-stroke-data.csv")

data['smoking_status'] = data['smoking_status'].replace('Unknown', np.nan)

data['ever_married'] = data['ever_married'].map({'Yes': 1, 'No': 0})
data['Residence_type'] = data['Residence_type'].map({'Urban': 1, 'Rural': 0})
data['smoking_status'] = data['smoking_status'].map({'smokes': 1, 'never smoked': 0, 'formerly smoked': 2, 'Unknown': 3})
data['work_type'] = data['work_type'].map({'self-employed': 1, 'govt_job': 2, 'Private': 3 ,'children': 4,'Never_worked': 5})
data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
 
# Supprimer les lignes contenant des valeurs NaN
data.dropna(inplace=True)

# Diviser les données en features et target
X = data.drop(columns=['id', 'stroke'])
y = data['stroke']

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X, y)
y_pred = cross_val_predict(dt_classifier, X, y, cv=10)

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
print("* DECISION TREE RESULTS (Cross Validation) :")
print("Accuracy:", accuracy)
print("Précision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

# Afficher l'arbre de décision
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=['No Stroke', 'Stroke'], filled=False, fontsize=5, max_depth=3)
plt.title("Decision Tree")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=['Stroke', 'No stroke'], yticklabels=['Stroke', 'No stroke'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()