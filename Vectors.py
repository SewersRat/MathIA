def mainMenu():
    print("1. Vecteurs")
    print("2. Matrices")
    print("3. Quitter")
    print("Veuillez choisir une option: ")
    choice = input()
    if choice == "1":
        vecteurs()
    elif choice == "2":
        matrices()
    elif choice == "3":
        print("Au revoir!")
    else:
        print("Option invalide, veuillez réessayer.")
        mainMenu()

def vecteurs():
    v1 = [5,6]
    v2 = [7,8]
    resultat = [v1[0] + v2[0], v1[1] + v2[1]]
    print(resultat)
    input("Appuyez sur Entrée pour revenir au menu principal...")
    mainMenu()

def matrices():

    v1 = [5,6]

    matrice = [[1,2],[3,4]]
    resultat = [matrice[0][0] * v1[0] + matrice[0][1] * v1[1], matrice[1][0] * v1[0] + matrice[1][1] * v1[1]]
    print(resultat)
    input("Appuyez sur Entrée pour revenir au menu principal...")
    mainMenu()

mainMenu()