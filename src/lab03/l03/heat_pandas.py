import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def print_statistics(df):
    for name in df.columns:
        column = df[name]
        print("\nZmienna:", column)
        print("MIN:", column.min())
        print("MAX:", column.max())
        print("ŚREDNIA:", column.mean())
        print("MEDIANA:", column.median())
        print("ZAKRES:", column.max() - column.min())
        print("ODCHYLENIE STANDARDOWE:", column.std())
        print("WARIANCJA:", column.var())
        print("PERCENTYL 90%:", column.quantile(0.9))

        column.plot.hist(bins=100)
        plt.title('Histogram dla: ' + name)
        plt.xlabel('Przedział')
        plt.ylabel('Liczba obserwacji')
        plt.show()


df = pd.read_csv('dane.csv')
rename_dict = {'przeplyw': 'Przepływ', 'temperatura_zasilania': 'Temperatura zasilania', 'temperatura_powrotu': 'Temperatura powrotu', 'roznica_temperatur': 'Różnica temperatur', 'moc': 'Moc'}
df.rename(columns=rename_dict, inplace=True)
df.drop('data_odczytu', axis=1, inplace=True)

### ZADANIE (0.5p.) ###
# Dane w listach są ułożone od najnowszych do najstarszych.
# Odwrócić dane na listach tak, żeby były ułożone chronologicznie.
df = df.iloc[::-1]

print("Before:")
print_statistics(df)

### ZADANIE (1p.) ###
# Zrealizować automatyczne dodawanie "podejrzanych" zmiennych do słownika "zmienne_do_naprawienia",
# na podstawie analizy statystyk danej zmiennej.
### ZADANIE (1p.) ###
# Znaleźć inną metodę wyznaczania progu anomalii w powyższej pętli tak, aby nie była to
# "hardkodowana" wartość 10000, ale liczba wyznaczana indywidualnie dla każdej zmiennej.
# Jedna z metod - metoda IQR: https://online.stat.psu.edu/stat200/lesson/3/3.2
# Inna podpowiedź: https://mateuszgrzyb.pl/3-metody-wykrywania-obserwacji-odstajacych-w-python/
print("Filtering...")
for name in df.columns:
    column = df[name]
    q1 = column.quantile(0.25)
    q3 = column.quantile(0.75)
    IQR = q3 - q1
    median = column.median()
    lower_fence = median - 1.5 * IQR
    upper_fence = median + 1.5 * IQR
    df[name] = df[name].apply(lambda x: median if x < lower_fence or upper_fence < x else x)

print("After:")
print_statistics(df)

### ZADANIE (1p.) ###
# Zapisać powyższe statystyki i wykresy do plików PDF, osobnych dla poszczególnych zmiennych
# (można wykorzystać dowolną metodę/moduł/bibliotekę/pakiet).
for name in df.columns:
    with PdfPages(f"{name}.pdf") as pdf:
        column = df[name]

        txt = []
        txt.append(f"MIN: {column.min()}")
        txt.append(f"MAX: {column.max()}")
        txt.append(f"ŚREDNIA: {column.mean()}")
        txt.append(f"MEDIANA: {column.median()}")
        txt.append(f"ZAKRES: {column.max() - column.min()}")
        txt.append(f"ODCHYLENIE STANDARDOWE: {column.std()}")
        txt.append(f"WARIANCJA: {column.var()}")
        txt.append(f"PERCENTYL 90%: {column.quantile(0.9)}")

        info = plt.figure()
        info.clf()
        info.text(0.1, 0.3, "\n".join(txt), fontsize=12)
        pdf.savefig()
        plt.close()
        
        column.plot.hist(bins=100)
        plt.title('Histogram dla: ' + name)
        plt.xlabel('Przedział')
        plt.ylabel('Liczba obserwacji')
        pdf.savefig()
        plt.close()

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

### ZADANIE (0.5p.) ###
# Poszukać funkcji z pakietu numpy, która wykonuje identyczne zadanie jak
# funkcja ncorrelate() i ją wykorzystać.
print(f"Correlated numpy example:")
print(f"Correlation score: {np.corrcoef(correlated1, correlated2)}")

for name1, column1 in df.items():
    for name2, column2 in df.items():
        if name1 != name2:
            print("Korelacja między", name1,"a", name2,"wynosi:", end=" ")
            print(ncorrelate(column1, column2))

### ZADANIE (1p.) ###
# Zebrać powyższe wyniki korelacji w dwuwymiarowej liście postaci:
# [[zmienna1, zmienna2, korelacja], [..., ..., ...], ... ] tak, aby elementy tej listy
# były posortowane malejąco wg. wartości korelacji.
correlation_list = []
for name1, column1 in df.items():
    for name2, column2 in df.items():
        if name1 != name2:
            correlation_list.append([name1, name2, ncorrelate(column1, column2)])
correlation_list = sorted(correlation_list, key=lambda x: x[2], reverse=True)
print(correlation_list)

moc = df["Moc"]
przeplyw = df["Przepływ"]
roznica_temp = df["Różnica temperatur"]
temp_powrotu = df["Temperatura powrotu"]
temp_zasilania = df["Temperatura zasilania"]
plt.plot(range(len(moc)), moc, "x", label="Moc")
plt.plot(range(len(przeplyw)), przeplyw, "+", label="Przepływ")
plt.title("Duża korelacja dodatnia")
plt.xlabel('Numer obserwacji')
plt.legend() 
plt.show()

# Dla lepszej ilustracji: wycinek danych.
# Zmienna moc przemnożnona przez 10, aby lepiej było widać korelację.
plt.plot(range(len(moc.iloc[1000:1100])), [i*10 for i in moc.iloc[1000:1100]], label="Moc")
plt.plot(range(len(przeplyw.iloc[1000:1100])), przeplyw.iloc[1000:1100], label="Przepływ")
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
plt.plot(range(len(roznica_temp.iloc[1000:1100])), roznica_temp.iloc[1000:1100], label="Różnica temperatur")
plt.plot(range(len(temp_powrotu.iloc[1000:1100])), temp_powrotu.iloc[1000:1100], label="Temperatura powrotu")
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
temp_zasilania_4_pory_roku = np.array_split(temp_zasilania, 4)
for temp_zasilania_pora_roku in temp_zasilania_4_pory_roku:
    x = range(len(temp_zasilania_pora_roku))
    y = temp_zasilania_pora_roku
    a,b = np.polyfit(x,y,1)  # Wielomian 1 rzędu - prosta
    print("Wzór prostej: y(x) =",a,"* x +",b)
    yreg =  [a*i + b for i in x]
    plt.plot(x,y, label="Temperatura zasilania")
    plt.plot(x,yreg, label="Wynik regresji")
    plt.title("Regresja liniowa dla części danych zmiennej temp_zasilania")
    plt.xlabel('Numer obserwacji')
    plt.legend()
    plt.show()

# Regresja liniowa dla zmiennych z dużą korelacją dodatnią: moc, przeplyw
a,b = np.polyfit(moc,przeplyw,1)  # Wielomian 1 rzędu - prosta
yreg =  [a*i + b for i in moc]
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
plt.plot(roznica_temp,temp_powrotu,".")
plt.plot(roznica_temp,yreg, label="Wynik regresji")
plt.title("Regresja liniowa")
plt.xlabel('Różnica temperatur')
plt.ylabel('Temperatura powrotu')
plt.legend()
plt.show()

roznica_temp = []
import random
for i in range(20):
	roznica_temp.append(random.randint(0,100))

temp_powrotu = [[i, a*i+b] for i in roznica_temp]
print("Wyniki predykcji temp_powrotu(roznica_temp):",temp_powrotu)

### ZADANIE (0.5p.) ###
import json
with open('prediction.json', 'w') as file:
    json.dump(temp_powrotu, file)
