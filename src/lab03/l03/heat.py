###############################################################################
#
# Python 3
#
# Wcięcia realizowane są czterema spacjami.
#
# Doinstalowanie bibliotek numpy i matplotlib:
# python -m pip install numpy
# python -m pip install matplotlib
#
# Uruchamianie skryptu:
# python heat.py
# albo wymuszając Pythona 3 gdy nie jest on domyślny:
# py -3 heat.py
#
###############################################################################
#
# Plik dane.csv zawiera dane zbierane na węźle ciepłowniczym przez
# przedsiębiorstwo dostarczające ciepło do budynku (patrz opisy kolumn w pliku).
# Niniejszy skrypt dokonuje podstawowej analizy tych danych.
#
# A.
# Wczytanie obserwacji z pliku.
#
# B.
# Sprawdzenie podstawowych statystyk dla poszczególnych zmiennych.
# Wykreślenie histogramów.
#
# C.
# Identyfikacja zmiennych, w których występują potencjalnie błędne dane (obserwacje)
# lub braki danych. Naprawa danych.
#
# D.
# Obliczenie unormowanych korelacji pomiędzy poszczególnymi zmiennymi.
#
# E.
# Przeprowadzenie regresji liniowej dla wybranych zmiennych, wraz z wykresami.
#
###############################################################################
#
# Należy wykonać zadania na sumarycznie co najmniej 4 punkty
#
###############################################################################


import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#######################
# A. Wczytanie danych #
#######################

przeplyw = []        # Przepływ wody przez węzeł
temp_zasilania = []  # Temperatura wody na wejściu do węzła
temp_powrotu = []    # Temperatura wody na wyjściu z węzła
roznica_temp = []    # Różnica temperatur, wynikająca z oddanej energii w węźle
moc = []             # Moc oddana w węźle

plik = open('dane.csv', 'rt')
dane = csv.reader(plik, delimiter=',')
next(dane)                # Opuszczamy pierwszy wiersz (nagłówek)
for obserwacja in dane:   # Iterujemy po poszczególnych obserwacjach
    przeplyw.append(float(obserwacja[1]))
    temp_zasilania.append(float(obserwacja[2]))
    temp_powrotu.append(float(obserwacja[3]))
    roznica_temp.append(float(obserwacja[4]))
    moc.append(float(obserwacja[5]))
plik.close()

### ZADANIE (0.5p.) ###
# Dane w listach są ułożone od najnowszych do najstarszych.
# Odwrócić dane na listach tak, żeby były ułożone chronologicznie.
przeplyw.reverse()
temp_zasilania.reverse()
temp_powrotu.reverse()
roznica_temp.reverse()
moc.reverse()
### KONIEC ###

# Tworzymy słownik: kluczem jest nazwa zmiennej a wartością - zmienna
zmienne = {"Przepływ":przeplyw,
           "Temperatura zasilania":temp_zasilania,
           "Temperatura powrotu":temp_powrotu,
           "Różnica temperatur":roznica_temp,
           "Moc":moc
           }


######################################
# B. Podstawowe statystyki i wykresy #
######################################

# Iterujemy po słowniku, wyświetlając statystyki dla poszczególnych zmiennych
for nazwa,zmienna in zmienne.items():
    print("\nZmienna:",nazwa)
    print("MIN:", min(zmienna))
    print("MAX:", max(zmienna))
    print("ŚREDNIA:", np.mean(zmienna))
    print("MEDIANA:", np.median(zmienna))
    print("ZAKRES:", np.ptp(zmienna))
    print("ODCHYLENIE STANDARDOWE:", np.std(zmienna))
    print("WARIANCJA:", np.var(zmienna))
    print("PERCENTYL 90%:", np.percentile(zmienna,90) )
    print("HISTOGRAM:", np.histogram(zmienna))

    # Wykres - histogram
    plt.hist(zmienna, 100)
    plt.title('Histogram dla: ' + nazwa)
    plt.xlabel('Przedział')
    plt.ylabel('Liczba obserwacji')
    plt.show()


############################################
# C. Analiza anomalii i czyszczenie danych #
############################################

# Zidentyfikowaliśmy problem - "dziwne", znacząco za duże niektóre wartości dla zmiennych:
zmienne_do_naprawienia = {"Przepływ":przeplyw,
                          "Różnica temperatur":roznica_temp,
                          "Moc":moc}

### ZADANIE (1p.) ###
# Zrealizować automatyczne dodawanie "podejrzanych" zmiennych do słownika "zmienne_do_naprawienia",
# na podstawie analizy statystyk danej zmiennej.
for nazwa, zmienna in zmienne.items():
    if np.max(zmienna) > 10000:
        zmienne_do_naprawienia[nazwa] = zmienna
### KONIEC ###

print("\nCZYSZCZENIE DANYCH")

for nazwa,zmienna in zmienne_do_naprawienia.items():
    for index,wartosc in enumerate(zmienna): # Iterujemy po wszystkich obserwacjach
        # Zakładamy (na podstawie analizy danych), że anomalia to wartość powyżej 10000
        if (wartosc > 10000):
            print("Dla zmiennej {} pod indeksem {} znaleziono anomalię o wartości {}".format(nazwa, index, wartosc))
            # Wstawiamy medianę:
            mediana = np.median(zmienna)
            print("Naprawiam. Stara wartość: {}, nowa wartość: {}".format(zmienna[index], mediana))
            zmienna[index] = mediana

### ZADANIE (1p.) ###
# Znaleźć inną metodę wyznaczania progu anomalii w powyższej pętli tak, aby nie była to
# "hardkodowana" wartość 10000, ale liczba wyznaczana indywidualnie dla każdej zmiennej.
# Jedna z metod - metoda IQR: https://online.stat.psu.edu/stat200/lesson/3/3.2
# Inna podpowiedź: https://mateuszgrzyb.pl/3-metody-wykrywania-obserwacji-odstajacych-w-python/
print("\nCZYSZCZENIE DANYCH")

for nazwa,zmienna in zmienne_do_naprawienia.items():
    q1 = np.percentile(zmienna, 25)
    q3 = np.percentile(zmienna, 75)
    IQR = q3 - q1
    IQR_3by2 = 1.5 * IQR
    mediana = np.median(zmienna)
    lower_fence = mediana - IQR_3by2
    upper_fence = mediana + IQR_3by2
    for index,wartosc in enumerate(zmienna):
        if wartosc < lower_fence or upper_fence < wartosc:
            print("Dla zmiennej {} pod indeksem {} znaleziono anomalię o wartości {}".format(nazwa, index, wartosc))
            print("Naprawiam. Stara wartość: {}, nowa wartość: {}".format(zmienna[index], mediana))
            zmienna[index] = mediana
### KONIEC ###

# Statystyki dla naprawionych zmiennych
for nazwa,zmienna in zmienne.items():
    print("\nZmienna (naprawiona):",nazwa)
    print("MIN:", min(zmienna))
    print("MAX:", max(zmienna))
    print("ŚREDNIA:", np.mean(zmienna))
    print("MEDIANA:", np.median(zmienna))
    print("ZAKRES:", np.ptp(zmienna))
    print("ODCHYLENIE STANDARDOWE:", np.std(zmienna))
    print("WARIANCJA:", np.var(zmienna))
    print("PERCENTYL 90%:", np.percentile(zmienna,90))
    print("HISTOGRAM:", np.histogram(zmienna))

    plt.hist(zmienna, 100)
    plt.title('Histogram dla: ' + nazwa)
    plt.xlabel('Przedział')
    plt.ylabel('Liczba obserwacji')
    plt.show()

### ZADANIE (1p.) ###
# Zapisać powyższe statystyki i wykresy do plików PDF, osobnych dla poszczególnych zmiennych
# (można wykorzystać dowolną metodę/moduł/bibliotekę/pakiet).
for nazwa,zmienna in zmienne.items():
    pdf_pages = PdfPages(f"output_{nazwa}.pdf")
    pdf_pages.attach_note("\nZmienna (naprawiona): {}\n".format(nazwa))
    pdf_pages.attach_note("MIN: {}\n".format(min(zmienna)))
    pdf_pages.attach_note("MAX: {}\n".format(max(zmienna)))
    pdf_pages.attach_note("ŚREDNIA: {}\n".format(np.mean(zmienna)))
    pdf_pages.attach_note("MEDIANA: {}\n".format(np.median(zmienna)))
    pdf_pages.attach_note("ZAKRES: {}\n".format(np.ptp(zmienna)))
    pdf_pages.attach_note("ODCHYLENIE STANDARDOWE: {}\n".format(np.std(zmienna)))
    pdf_pages.attach_note("WARIANCJA: {}\n".format(np.var(zmienna)))
    pdf_pages.attach_note("PERCENTYL 90%: {}\n".format(np.percentile(zmienna, 90)))

    plt.hist(zmienna, 100)
    plt.title('Histogram dla: ' + nazwa)
    plt.xlabel('Przedział')
    plt.ylabel('Liczba obserwacji')
    pdf_pages.savefig()
    pdf_pages.close()
### KONIEC ###


#########################################
# D. Badanie korelacji między zmiennymi #
#########################################

print()
print("KORELACJE")

# Piszemy funkcję, która zwróci korelację unormowaną między zestawami danych
def ncorrelate(a,b):
    '''Funkcja zwraca unormowaną wartość korelacji'''
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / np.std(b)
    return np.correlate(a, b)[0]

### ZADANIE (0.5p.) ###
# Zademonstrować działanie funkcji ncorrelate() na przykładach:
# a. dwóch list zawierających dane silnie skorelowane
# b. dwóch list zawierających dane słabo skorelowane
# Listy należy generować automatycznie
correlated1 = [i for i in range(0, 20, 2)]
correlated2 = [i for i in range(0, 40, 4)]
print(f"Correlated example (Przepływ and Temperatura zasilania):")
print(f"Correlation score: {ncorrelate(correlated1, correlated2)}")

import random
random.seed(0)
not_correlated1 = [random.random() for _ in range(20)]
not_correlated2 = [random.random() for _ in range(20)]
print(f"Not correlated example:")
print(f"Correlation score: {ncorrelate(not_correlated1, not_correlated2)}")
### KONIEC ###

### ZADANIE (0.5p.) ###
# Poszukać funkcji z pakietu numpy, która wykonuje identyczne zadanie jak
# funkcja ncorrelate() i ją wykorzystać.
print(f"Correlated numpy example:")
print(f"Correlation score: {np.corrcoef(correlated1, correlated2)}")
### KONIEC ###

# Badamy korelacje między wszystkimi (różnymi od siebie) zmiennymi
for nazwa1,zmienna1 in zmienne.items():
    for nazwa2,zmienna2 in zmienne.items():
        if nazwa1 != nazwa2:
            print("Korelacja między", nazwa1,"a", nazwa2,"wynosi:", end=" ")
            print(ncorrelate(zmienna1,zmienna2))

### ZADANIE (1p.) ###
# Zebrać powyższe wyniki korelacji w dwuwymiarowej liście postaci:
# [[zmienna1, zmienna2, korelacja], [..., ..., ...], ... ] tak, aby elementy tej listy
# były posortowane malejąco wg. wartości korelacji.
correlation_list = []
for nazwa1,zmienna1 in zmienne.items():
    for nazwa2,zmienna2 in zmienne.items():
        if nazwa1 != nazwa2:
            correlation_list.append([nazwa1, nazwa2, ncorrelate(zmienna1, zmienna2)])
print(correlation_list)
### KONIEC ###

# Przykładowe wykresy

# 1. Zmienne z dużą korelacją dodatnią: moc, przeplyw

# Wykres liniowy
plt.plot(range(len(moc)), moc, "x", label="Moc")
plt.plot(range(len(przeplyw)), przeplyw, "+", label="Przepływ")
plt.title("Duża korelacja dodatnia")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Dla lepszej ilustracji: wycinek danych.
# Zmienna moc przemnożnona przez 10, aby lepiej było widać korelację.
plt.plot(range(len(moc[1000:1100])), [i*10 for i in moc[1000:1100]], label="Moc")
plt.plot(range(len(przeplyw[1000:1100])), przeplyw[1000:1100], label="Przepływ")
plt.title("Duża korelacja dodatnia; Moc przemnożona przez 10")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Wykres zależności przeplyw od moc
plt.plot(moc, przeplyw, '.')
plt.title("Duża korelacja dodatnia")
plt.xlabel('Moc')
plt.ylabel('Przeplyw')
plt.show()

# 2. Zmienne skorelowane ujemnie: roznica_temp, temp_powrotu

# Wykres liniowy
plt.plot(range(len(roznica_temp)), roznica_temp, "x", label="Różnica temperatur")
plt.plot(range(len(temp_powrotu)), temp_powrotu, "+", label="Temperatura powrotu")
plt.title("Średnia korelacja ujemna")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Dla lepszej ilustracji: wycinek danych
plt.plot(range(len(roznica_temp[1000:1100])), roznica_temp[1000:1100], label="Różnica temperatur")
plt.plot(range(len(temp_powrotu[1000:1100])), temp_powrotu[1000:1100], label="Temperatura powrotu")
plt.title("Średnia korelacja ujemna")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

# Wykres zależności temp_powrotu od roznica_temp
plt.plot(roznica_temp, temp_powrotu, '.')
plt.title("Średnia korelacja ujemna")
plt.xlabel('Różnica temperatur')
plt.ylabel('Temperatura powrotu')
plt.show()


#######################
# E. Regresja liniowa #
#######################

# Analiza przeprowadzona tylko dla jednej zmiennej, temp_zasilania

print()
print("REGRESJA LINIOWA")
# Wybieramy zmienną temp_zasilania w funkcji numeru obserwacji
x = range(len(temp_zasilania))
y = temp_zasilania
# Liczymy współczynniki regresji - prostej
a,b = np.polyfit(x,y,1)  # Wielomian 1 rzędu - prosta
print("Wzór prostej: y(x) =",a,"* x +",b)
# Wyliczamy punkty prostej otrzymanej w wyniku regresji
yreg =  [a*i + b for i in x]
# Wykresy
plt.plot(x,y, label="Temperatura zasilania")
plt.plot(x,yreg, label="Wynik regresji")
plt.title("Regresja liniowa dla całosci danych zmiennej temp_zasilania")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()


print()
print("REGRESJA WIELOMIANOWA")
# Wybieramy zmienną temp_zasilania w funkcji numeru obserwacji
x = range(len(temp_zasilania))
y = temp_zasilania
# Liczymy współczynniki regresji - prostej
a,b,c = np.polyfit(x,y,2)  # Wielomian 1 rzędu - prosta
print("Wzór krzywej: y(x) =",a,"* x *x  +",b,"* x +",c)
# Wyliczamy punkty prostej otrzymanej w wyniku regresji
yreg =  [a*i*i + b*i + c for i in x]
# Wykresy
plt.plot(x,y, label="Temperatura zasilania")
plt.plot(x,yreg, label="Wynik regresji")
plt.title("Regresja liniowa dla całosci danych zmiennej temp_zasilania")
plt.xlabel('Numer obserwacji')
plt.legend()
plt.show()

### ZADANIE (1.5p.) ###
# Z wykresu widać, że regresja liniowa dla całości zmiennej temp_zasilania słabo się sprawdza.
# Wynika to z tego, że inaczej dane rozkładają się w róznych porach roku.
# Należy więc podzielić dane na kilka podzakresów i regresję wykonać osobno
# dla każdego z podzakresu. Narysować odpowiedni wykres.
### KONIEC ###

# Regresja liniowa dla zmiennych z dużą korelacją dodatnią: moc, przeplyw
a,b = np.polyfit(moc,przeplyw,1)  # Wielomian 1 rzędu - prosta
yreg =  [a*i + b for i in moc]
# Wykresy
plt.plot(moc,przeplyw,".")
plt.plot(moc,yreg, label="Wynik regresji")
plt.title("Regresja liniowa")
plt.xlabel('Moc')
plt.ylabel('Przepływ')
plt.legend()
plt.show()

# Regresja liniowa dla zmiennych ze słabą korelacją ujemną: roznica_temp, temp_powrotu
a,b = np.polyfit(roznica_temp,temp_powrotu,1)  # Wielomian 1 rzędu - prosta
yreg =  [a*i + b for i in roznica_temp]
# Wykresy
plt.plot(roznica_temp,temp_powrotu,".")
plt.plot(roznica_temp,yreg, label="Wynik regresji")
plt.title("Regresja liniowa")
plt.xlabel('Różnica temperatur')
plt.ylabel('Temperatura powrotu')
plt.legend()
plt.show()

# Predykcja danych z losowej listy
roznica_temp = []
import random
for i in range(20):
	roznica_temp.append(random.randint(0,100))

# Wyliczenie wyników na podstawie regresji i zapis do listy
temp_powrotu = [[i, a*i+b] for i in roznica_temp]
print("Wyniki predykcji temp_powrotu(roznica_temp):",temp_powrotu)

### ZADANIE (0.5p.) ###
# Zapisać wyniki powyższej predykcji do pliku JSON o nazwie predykcja.json
### KONIEC ###
