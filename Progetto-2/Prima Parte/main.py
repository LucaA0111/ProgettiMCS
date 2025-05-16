from dctFunction import compare_dct
from verify_dct import verify_dct2
from verify_dct import verify_dct1


def main():
    # Chiedi all'utente di inserire le dimensioni della matrice
    user_input = input("Inserisci le dimensioni della matrice separando i valori con uno spazio: ")

    # Converte l'input dell'utente in una lista di numeri interi
    n_values = list(map(int, user_input.split()))

    # Esegui il confronto
    compare_dct(n_values)


# Esegui il programma
if __name__ == "__main__":
    if verify_dct1():
        if verify_dct2():
            main()
        else:
            print("Errore: La funzione DCT2 non ha superato la verifica.")
    else:
        print("Errore: La funzione DCT1 non ha superato la verifica.")
