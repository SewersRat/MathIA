# TP final – Neurone multivarié avec MSE et Gradient Descent

import numpy as np

# ---- Données ----
# 5 exemples avec 2 features chacun (x1, x2)
# Le modèle cible est y = 3*x1 - 2*x2 + 1
# => on s'attend à apprendre w = [3, -2] et b = 1

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [2, 1]
])

y = np.array([1, -1, 4, 2, 6])


# ---- Exercice 1 – Prédiction du neurone ----
# Un neurone linéaire calcule une combinaison linéaire des entrées :
# y_pred = w1*x1 + w2*x2 + b, ce qui s'écrit x·w + b avec le produit scalaire

def predict(x, w, b):
    """
    x : vecteur des entrées [x1, x2]
    w : vecteur des poids [w1, w2]
    b : biais
    """
    y_pred = np.dot(x, w) + b  # produit scalaire + biais
    return y_pred


# ---- Exercice 2 – Calcul de la MSE ----
# La MSE mesure l'écart moyen entre les prédictions et les vraies valeurs
# Plus elle est proche de 0, meilleur est le modèle
# On élève au carré pour pénaliser les grandes erreurs et éviter les annulations

def mse(y_true, y_pred):
    error = np.mean((y_true - y_pred) ** 2)  # moyenne des carrés des erreurs
    return error


# ---- Exercice 3 – Gradient Descent ----
# On part de w=[0,0] et b=0 (modèle nul), puis on ajuste à chaque étape
# Le gradient indique dans quelle direction la MSE augmente
# => on fait le contraire (on soustrait) pour minimiser l'erreur
# lr (learning rate) contrôle la taille du pas : trop grand = diverge, trop petit = lent

def train(X, y, lr=0.01, epochs=200):
    w = np.array([0.0, 0.0])  # initialisation des poids à 0
    b = 0.0
    n = len(X)

    for i in range(epochs):
        y_pred = np.array([predict(x, w, b) for x in X])

        # gradient de MSE par rapport à w : moyenne de (erreur * entrée)
        # si dw[0] > 0, w[0] est trop grand => on le diminue
        dw = (2/n) * np.dot(X.T, (y_pred - y))

        # gradient par rapport à b : juste la moyenne des erreurs
        db = (2/n) * np.sum(y_pred - y)

        w = w - lr * dw  # on se déplace dans le sens opposé au gradient
        b = b - lr * db

    return w, b


# ---- Exercice 4 – Test du modèle ----
# 1000 epochs suffisent pour converger sur des données aussi simples
# Avec lr=0.01 et des données linéaires, le modèle devrait trouver
# les vrais paramètres : w ≈ [3, -2], b ≈ 1

w, b = train(X, y, lr=0.01, epochs=1000)

print("Poids appris :", w)   # attendu : [3. -2.]
print("Biais appris :", b)   # attendu : 1.0

# Prédiction sur une nouvelle donnée non vue à l'entraînement
# valeur attendue : 3*3 - 2*2 + 1 = 6
x_test = np.array([3, 2])
print("Prediction pour x_test =", predict(x_test, w, b))
# Si le résultat est proche de 6, le modèle a bien généralisé