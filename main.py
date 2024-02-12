import pandas as pd

# Chemin vers le fichier CSV
chemin_fichier = "./data_sets/bodyPerformance.csv"

# Charger le fichier CSV dans un DataFrame
data = pd.read_csv(chemin_fichier)
print(data.columns)

with pd.option_context('display.max_rows', 5, 'display.max_columns', None):
    print(data)

data_with_cat = data
data_with_cat['class'] = data_with_cat['class'].astype('category').cat.codes
data_with_cat['gender'] = data_with_cat['gender'].astype('category').cat.codes
#
# # Calculer la matrice de corrélation
# correlation_matrix = data_with_cat.corr()
#
# # Afficher la matrice de corrélation avec seaborn
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
# plt.title("Matrice de corrélation entre les variables physiques")
# plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Étape 1 : Sélectionner les caractéristiques paramètre/résultat
# features = ['sitand_bend_forward_cm', 'sit-ups counts', 'body fat_%']
features = ['age', 'gender', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic',
            'systolic', 'gripForce', 'sitand_bend_forward_cm', 'sit-ups counts',
            'broad jump_cm']
X = data_with_cat[features]
y = data_with_cat['class']

# Étape 2 : Diviser les données en jeu d'apprentissage/jeu de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Étape 3 : Choisir un algorithme et entraîner le modèle
model = LinearRegression()  # Mauvais algorithme d'apprentissage pour ce cas
model.fit(X_train, y_train)

# Étape 4 : Évaluer le modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print('Score pas très pertinent ici, mais on voit que la précision n\'est pas très bonne ni trop mauvaise')

# On recommence avec un cas plus simple pas pondu par ChatGPT pour mieux comprendre

# Étape 1 : Sélectionner les caractéristiques
# features = ['sitand_bend_forward_cm', 'sit-ups counts', 'body fat_%']
features2 = ['age', 'gender', 'class']
X2 = data_with_cat[features2]
y2 = data_with_cat['broad jump_cm']

# Étape 2 : Diviser les données
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=2)

# Étape 3 : Choisir un algorithme et entraîner le modèle
model2 = LinearRegression()
model2.fit(X2_train, y2_train)

# Étape 4 : Évaluer le modèle
y2_pred = model2.predict(X2_test)
mse2 = mean_squared_error(y2_test, y2_pred)
print(f'Mean Squared Error: {mse2}')
print('On a la distance de saut à ~20cm près en moyenne, ce qui est pas mal par rapport à la répartition')