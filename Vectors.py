# Selecteur, 2026-03-10
def mainMenu():
    print("1. Vecteurs")
    print("2. Matrices")
    print("3. Moyenne de deux vecteurs")
    print("4. Quitter")
    print("Veuillez choisir une option: ")
    choice = input()
    if choice == "1":
        vecteurs()
    elif choice == "2":
        matrices()
    elif choice == "3":
        Moyenne()
    elif choice == "4":
        print("Au revoir!")
    else:
        print("Option invalide, veuillez réessayer.")
        mainMenu()

#exo 1
def vecteurs():
    v1 = [5,6]
    v2 = [7,8]
    resultat = [v1[0] + v2[0], v1[1] + v2[1]]
    print(resultat)
    input("Appuyez sur Entrée pour revenir au menu principal...")
    mainMenu()
#exo 2
def matrices():

    v1 = [5,6]

    matrice = [[1,2],[3,4]]
    resultat = [matrice[0][0] * v1[0] + matrice[0][1] * v1[1], matrice[1][0] * v1[0] + matrice[1][1] * v1[1]]
    print(resultat)
    input("Appuyez sur Entrée pour revenir au menu principal...")
    mainMenu()
#Exo 3
def Moyenne():
    x = [50, 70, 90]
    y = [100, 140, 180]

    n = len(x)

    x_bar = sum(x) / n
    y_bar = sum(y) / n

    numerateur = 0
    denominateur = 0

    for i in range(n):
        numerateur += (x[i] - x_bar) * (y[i] - y_bar)
        denominateur += (x[i] - x_bar) ** 2

    a = numerateur / denominateur
    b = y_bar - a * x_bar

    print("a =", a)
    print("b =", b)
    print("Equation : y =", a, "x +", b)
    input("Appuyez sur Entrée pour revenir au menu principal...")
    mainMenu()

    
    

mainMenu()