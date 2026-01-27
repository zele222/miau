Blok 1: Dane studenta


Blok 2: Import bibliotek i wczytanie danych

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, morphology, util

# Funkcja pomocnicza do wyświetlania
def pokaz(im, tytul="", osie=False):
    if not(osie):
        plt.axis("off") 
    if im.ndim == 2:
        plt.imshow(im, cmap='gray')
    else:
        plt.imshow(im)
    plt.title(tytul)
    plt.show()

# Wczytanie obrazu
# Zakładamy, że plik znajduje się w katalogu roboczym
image_path = 'obiekty2.tif' 
try:
    image = io.imread(image_path)
    pokaz(image, "Obraz oryginalny")
except FileNotFoundError:
    print(f"Błąd: Nie znaleziono pliku {image_path}. Upewnij się, że plik jest wgraną.")




Blok 3: Zadanie 1 - Segmentacja (5 pkt)
# --- ZADANIE 1: SEGMENTACJA ---

# 1. Konwersja do odcieni szarości (jeśli obraz jest kolorowy)
if image.ndim == 3:
    image_gray = color.rgb2gray(image)
else:
    image_gray = image

# 2. Progowanie metodą Otsu
thresh = filters.threshold_otsu(image_gray)
binary = image_gray > thresh

# 3. Sprawdzenie tła (jeśli rogi są białe, odwracamy obraz, aby obiekty były białe/True)
# Sprawdzamy średnią wartość pikseli w narożnikach
corners = [binary[0,0], binary[0,-1], binary[-1,0], binary[-1,-1]]
if np.mean(corners) > 0.5:
    binary = ~binary # Inwersja logiczna

# Wyświetlenie wyniku
plt.figure(figsize=(6, 6))
pokaz(binary, "Obraz binarny po segmentacji")



Blok 4: Zadanie 2 - Filtracja i Etykietowanie (5 pkt)

# --- ZADANIE 2: FILTRACJA I ETYKIETOWANIE ---

# 1. Filtracja (usuwanie drobnych szumów - morfologiczne otwarcie)
# Używamy dysku jako elementu strukturalnego
selem = morphology.disk(2)
binary_cleaned = morphology.opening(binary, selem)

# 2. Etykietowanie obiektów
label_image = measure.label(binary_cleaned)
num_objects = label_image.max()

print(f"Liczba wykrytych obiektów: {num_objects}")

# Wizualizacja etykiet (kolorowa mapa)
image_label_overlay = color.label2rgb(label_image, image=image_gray, bg_label=0)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
pokaz(binary_cleaned, "Obraz po filtracji")
plt.subplot(1, 2, 2)
pokaz(image_label_overlay, f"Zetykietowane obiekty: {num_objects}")



Blok 5: Zadanie 3 - Analiza Cech (5 pkt)
# --- ZADANIE 3: ANALIZA CECH ---

# 1. Wyznaczenie cech
# Wybieramy: area (pole), perimeter (obwód) oraz euler_number (topologia - do wykrywania otworów)
props = measure.regionprops_table(label_image, properties=('label', 'area', 'perimeter', 'euler_number', 'eccentricity'))

# 2. Utworzenie ramki danych (DataFrame)
df = pd.DataFrame(props)

# 3. Wyświetlenie podstawowych statystyk
print("Statystyki opisowe wybranych cech:")
print(df[['area', 'perimeter', 'eccentricity']].describe())

# 4. Wizualizacja zależności (Pairplot)
# Wykres parowy dla wybranych cech geometrycznych
plt.figure(figsize=(8, 6))
sns.pairplot(df, vars=['area', 'perimeter', 'eccentricity'])
plt.suptitle("Analiza eksploracyjna cech", y=1.02)
plt.show()



Blok 6: Zadanie 4 - Klasyfikacja (Otwory) (5 pkt)

# --- ZADANIE 4: KLASYFIKACJA (Z OTWOREM / BEZ OTWORU) ---

# Cecha rozróżniająca: Euler Number (Liczba Eulera)
# Euler = 1 -> Obiekt pełny (bez otworów)
# Euler <= 0 -> Obiekt z otworem (1 - liczba_otworów)

# Przygotowanie pustego obrazu RGB
output_image = np.zeros((*binary_cleaned.shape, 3))

# Listy do zliczania
count_holes = 0
count_no_holes = 0

# Pobranie właściwości regionów do iteracji
regions = measure.regionprops(label_image)

for region in regions:
    # Pobranie współrzędnych pikseli danego obiektu
    coords = region.coords
    
    # Sprawdzenie warunku (czy ma otwór)
    if region.euler_number < 1:
        # Obiekt z otworem -> KOLOR ZIELONY [0, 1, 0]
        output_image[coords[:, 0], coords[:, 1]] = [0, 1, 0]
        count_holes += 1
    else:
        # Obiekt bez otworu -> KOLOR CZERWONY [1, 0, 0]
        output_image[coords[:, 0], coords[:, 1]] = [1, 0, 0]
        count_no_holes += 1

# Wyniki liczbowe
print(f"Liczba obiektów z otworem (ZIELONE): {count_holes}")
print(f"Liczba obiektów bez otworu (CZERWONE): {count_no_holes}")

# Wyświetlenie obrazu wynikowego
plt.figure(figsize=(8, 8))
plt.imshow(output_image)
plt.title(f"Klasyfikacja: Zielone (z otworem) vs Czerwone (bez)")
plt.axis('off')
plt.show()