# Progetto 2 - Compressione di Immagini tramite la DCT

## Corso: Metodi del Calcolo Scientifico - AA 2024-2025

### Descrizione
Lo scopo di questo progetto è utilizzare l'implementazione della DCT2 in un ambiente open source e studiare gli effetti di un algoritmo di compressione tipo JPEG (senza matrice di quantizzazione) su immagini in toni di grigio. Il progetto prevede sia lo sviluppo di codice che la scrittura di una relazione.

---

## Parte 1 - Confronto degli algoritmi DCT2

### Obiettivo
- Implementare la DCT2 "fatta in casa" in un ambiente open source.
- Confrontare i tempi di esecuzione con la DCT2 della libreria (presumibilmente ottimizzata tramite FFT).

### Attività
- Utilizzare array quadrati N × N con N crescente.
- Tracciare un grafico (scale semilogaritmica sulle ordinate) dei tempi di esecuzione al variare di N per entrambi gli algoritmi.
- Aspettarsi una complessità O(N³) per l’algoritmo "fatto in casa" e O(N² log N) per quello della libreria.
- Inserire un breve resoconto preliminare nella relazione.

---

## Parte 2 - Software di Compressione

### Funzionalità richieste
- Creare un'interfaccia grafica per:
  - Caricare un'immagine `.bmp` in scala di grigi.
  - Selezionare:
    - `F`: dimensione dei macro-blocchi (finestrelle) per la DCT2.
    - `d`: soglia di taglio delle frequenze (intero tra 0 e 2F−2).

### Algoritmo
1. Suddividere l'immagine in blocchi F × F, partendo dall'alto a sinistra.
2. Per ogni blocco:
   - Calcolare la DCT2 (usando la libreria).
   - Eliminare i coefficienti con k + ℓ ≥ d.
   - Applicare l'IDCT2 al blocco modificato.
   - Arrotondare i valori ottenuti, ponendo a 0 i valori negativi e a 255 quelli superiori a 255.
3. Ricomporre l'immagine finale.
4. Visualizzare affiancate l'immagine originale e quella compressa.

---

## Requisiti
- Il software **deve essere open source** (ad esempio, Python con NumPy/SciPy, ma MATLAB è escluso).
- Se la libreria offre solo la DCT monodimensionale, si può applicare prima alle righe, poi alle colonne.

---

## Test di Verifica
### Blocco 8×8 di test:
```text
231  32 233 161  24  71 140 245
247  40 248 245 124 204  36 107
234 202 245 167   9 217 239 173
193 190 100 167  43 180   8  70
 11  24 210 177  81 243   8 112
 97 195 203  47 125 114 165 181
193  70 174 167  41  30 127 245
 87 149  57 192  65 129 178 228

