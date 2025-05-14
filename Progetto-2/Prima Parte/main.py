from dctFunction import compare_dct


def main():
    # Chiedi all'utente di inserire le dimensioni della matrice
    user_input = input("Inserisci le dimensioni della matrice separando i valori con uno spazio (es. 4 8 16 32 64): ")

    # Converte l'input dell'utente in una lista di numeri interi
    n_values = list(map(int, user_input.split()))

    # Esegui il confronto
    compare_dct(n_values)


# Esegui il programma
if __name__ == "__main__":
    main()
