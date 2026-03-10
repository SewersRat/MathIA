# Regression lineaire — Documentation mathematique et technique

## Objectif

Ce projet implemente une regression lineaire simple sur quatre jeux de donnees issus du quartet d'Anscombe. L'objectif est de trouver, pour chaque jeu, la droite qui ajuste au mieux les donnees, puis d'evaluer la qualite de cet ajustement a l'aide d'une fonction de cout.

---

## 1. La fonction de prediction parametree

### Definition

Une **fonction de prediction parametree** est une fonction mathematique dont le comportement depend de parametres reglables. Dans le cas de la regression lineaire simple, cette fonction prend la forme :

$$\hat{y} = a \cdot x + b$$

- $x$ : variable d'entree (variable explicative)
- $\hat{y}$ : valeur predite par le modele
- $a$ : pente de la droite (parametre directeur)
- $b$ : ordonnee a l'origine (biais)

### Role des parametres

Les parametres $a$ et $b$ definissent entierement la droite. Faire varier ces parametres revient a explorer l'espace de toutes les droites possibles. L'apprentissage consiste precisement a trouver les valeurs de $a$ et $b$ qui minimisent l'ecart entre les predictions $\hat{y}_i$ et les valeurs reelles $y_i$.

### Calcul analytique des parametres optimaux

La methode des moindres carres fournit une solution exacte. Les parametres optimaux sont :

$$a = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

$$b = \bar{y} - a \cdot \bar{x}$$

Avec $\bar{x}$ et $\bar{y}$ les moyennes respectives de $x$ et $y$.

### Correspondance dans le code

```python
moyenne_x = sum(x) / len(x)
moyenne_y = sum(y) / len(y)

numerateur = sum((x[i] - moyenne_x) * (y[i] - moyenne_y) for i in range(len(x)))
denominateur = sum((x[i] - moyenne_x) ** 2 for i in range(len(x)))

a = numerateur / denominateur
b = moyenne_y - a * moyenne_x
```

Chaque terme de la somme correspond directement a la formule mathematique. Le `numerateur` calcule la covariance non normalisee entre $x$ et $y$, et le `denominateur` calcule la variance non normalisee de $x$.

---

## 2. La fonction de cout — MSE

### Definition

La **Mean Squared Error** (erreur quadratique moyenne) mesure l'ecart moyen entre les valeurs predites par le modele et les valeurs reelles observees. Elle est definie par :

$$MSE = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2$$

Avec :
- $n$ : nombre d'observations
- $y_i$ : valeur reelle de la $i$-eme observation
- $\hat{y}_i = a \cdot x_i + b$ : valeur predite par le modele

### Proprietes

| Propriete | Description |
|---|---|
| Toujours positive ou nulle | Le carre empeche les erreurs positives et negatives de se compenser |
| Nulle si et seulement si | Le modele predit parfaitement toutes les observations |
| Sensible aux valeurs aberrantes | Le carre amplifie les grandes erreurs |
| Meme unite que $y^2$ | A interpreter dans le contexte des donnees |

### Interpretation pratique

Une MSE faible indique que les predictions sont proches des valeurs reelles. Elle ne peut s'interpreter qu'en valeur relative : comparer la MSE de deux modeles sur les memes donnees permet de determiner lequel ajuste mieux. Une MSE proche de zero signifie que la droite passe au plus pres de l'ensemble des points.

### Correspondance dans le code

```python
MSE = sum((y[i] - (a * x[i] + b)) ** 2 for i in range(len(x))) / len(x)
```

- `(a * x[i] + b)` calcule $\hat{y}_i$, la prediction pour le point $i$
- `(y[i] - (a * x[i] + b)) ** 2` calcule le carre de l'erreur pour ce point
- La somme est divisee par `len(x)` pour obtenir la moyenne

---

## 3. Interpretation des resultats

### Lecture de la droite de regression

Une fois $a$ et $b$ calcules, l'equation $y = a \cdot x + b$ peut s'interpreter directement :

- **$a > 0$** : relation positive entre $x$ et $y$ — quand $x$ augmente, $y$ augmente
- **$a < 0$** : relation negative — quand $x$ augmente, $y$ diminue
- **$|a|$** : amplitude de la variation de $y$ pour une unite de $x$
- **$b$** : valeur predite de $y$ quand $x = 0$

### Le quartet d'Anscombe

Les quatre jeux de donnees implementes dans ce projet sont concus pour avoir des statistiques descriptives quasi-identiques (meme moyenne, meme variance, meme $a$, meme $b$, meme MSE) tout en representant des structures completement differentes. Cela illustre que la MSE seule ne suffit pas a valider un modele lineaire — la visualisation graphique reste indispensable.

| Dataset | Structure reelle |
|---|---|
| 1 | Relation lineaire classique, ajustement coherent |
| 2 | Relation non lineaire (courbe), la droite est inappropriee |
| 3 | Relation lineaire parfaite perturbee par un point aberrant |
| 4 | $x$ constant sauf un point, la droite n'a aucun sens |

### Limites du modele lineaire

La regression lineaire suppose que la relation entre $x$ et $y$ est lineaire et que les residus $(y_i - \hat{y}_i)$ sont distribues de maniere aleatoire autour de zero. Si ces hypotheses ne sont pas verifiees, la MSE peut etre satisfaisante tout en cachant un modele inadapte.
