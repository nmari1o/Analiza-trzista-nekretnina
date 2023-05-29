
import pandas as pd
from statistics import mode

#spremanje podataka u Data Frame
df = pd.read_excel('apartments.xlsx')


import numpy as np

#spremanje vrijednosti (stupaca) iz Data Frame-a u numpy nizove
lokacija = np.array(df['Lokacija'].to_numpy())
cijena = np.array(df['Cijena'].to_numpy())
kvadratura = np.array(df['Kvadratura'].to_numpy())

#mode je vrijednost koja se najviše ponavlja u skupu vrijednosti
mode=mode(lokacija)
print("Najčešća lokacija za iznajmljivanje stanova: ", mode)


min=np.min(cijena)
print("Najmanja cijena najma stana: ",min,"€")

max=np.max(cijena)
print("Najveca cijena najma stana: ", max,"€")


#izračunavanje aritmetičke sredine svih kvadratura (prosjek kvadrata)
mean=np.mean(kvadratura)
print("Prosjek kvadratura: ",mean)

#Medijana predstavlja srednju vrijednost skupa podataka koja se nalazi točno u sredini kada su svi podaci sortirani po veličini. To znači da polovica podataka ima vrijednosti veće od medijane, a druga polovica ima vrijednosti manje od medijane.
median=np.median(cijena)
print("Medijan cijena: ", median)


#Standardna devijacija je mjera raspršenosti skupa podataka, tj. mjera koliko su podaci raspršeni u odnosu na srednju vrijednost. Ona pokazuje koliko se vrijednosti u skupu podataka razlikuju od srednje vrijednosti.
std = np.std(cijena)
print("Standardna devijacija cijena: ",std)

#Varijanca - mjera srednje kvadratne udaljenost pojedinačnih vrijednosti od aritmetičke sredine skupa podataka.
var=np.var(kvadratura)
print("Varijanca kvadratura: ",var)


# Definiranje DataFrame s podacima
data = pd.DataFrame({'Kvadratura': kvadratura,
'Cijena': cijena})

# Izračunavanje korelacije
corr = data['Kvadratura'].corr(data['Cijena'])
print()
print("Korelacija kvadrature i cijene najma: ", corr)




print()
# Provjera duplikata
duplicates = df.duplicated(keep=False)
print("Duplikati:\n", df[duplicates])
print()


#brisanje duplikata
df = df.drop_duplicates()
print()

#provjera brisanja duplikata
duplicates = df.duplicated(keep=False)
print("Duplikati:\n", df[duplicates])
print()



#ispitivanje dimenzija skupa podataka
print("Dimenzije skupa podataka: ", df.shape)

#ispitivanje veličine skupa podataka
print("Veličina skupa podataka: ", df.size)

#ispitivanje broja dimenzija skupa podataka
print("Broj dimenzija skupa podataka: ", df.ndim)

#ispis prvih nekoliko redaka
print("Prvih nekoliko redaka:")
print(df.head())

#ispis posljednjih nekoliko redaka
print("Posljednjih nekoliko redaka:")
print(df.tail())


print()
print()

#izdvajanje podskupa podataka s lokacijom 'Trešnjevka'
df_tresnjevka = df.loc[df['Lokacija'] == 'Trešnjevka']

#ispis prvih pet redaka izdvojenog podskupa podataka
print(df_tresnjevka.head())

#ispis stanova ciji je najam veci od 1000 eura
print()
subset = df.loc[df['Cijena'] > 1000]
print(subset)

#ispis stanova koji su velicine vise od 100 m^2
print()
subset = df.loc[(df['Kvadratura'] >= 100)]
print(subset)

print()
df.loc[df['Lokacija'] == ' Maksimir', 'Lokacija'] = 'Maksimir'


# Grupiranje podataka po lokaciji
grouped_by_location = df.groupby('Lokacija')

# Izračunavanje srednje vrijednosti cijene i kvadrature za svaku grupu
mean_by_location = grouped_by_location.mean()[['Cijena', 'Kvadratura']]

# Ispis rezultata
print(mean_by_location)


#sortiranje podataka po cijeni u silaznom redoslijedu
df.sort_values('Cijena', ascending=False, inplace=True)
print(df)


print()
print()
print()

#INFERENCIJALNA STATISTIČKA ANALIZA
#Hipoteza: pretpostavimo da je prosječna cijena najma stana u Zagrebu manja od 1000 Eura.  --> koristit će se jedostrani t-test
import scipy
from scipy import stats
from scipy.stats import ttest_ind, ttest_1samp

# postavljanje hipoteza
# H0: prosječna cijena >= 1000
# H1: prosječna cijena < 1000
# Izvlačenje cijena najma
cijene = df['Cijena']

# Jednostrani t-test
t_statistika, p_vrijednost = ttest_1samp(cijene, 1000, alternative='less')

# Ispis rezultata
print('t_statistika =', t_statistika)
print('p_vrijednost =', p_vrijednost) # jednaka 0.0067, dakle < 0.05, odbacujemo H0 i prihvaćamo H1 (prosječna cijena manja od 1000 EURA)



print()
print()
print()


#Neupareni t-test: Usporedba prosječne cijene najma u Gornjem Gradu i u Donjem Gradu
#H0: nema statisstički značajne razlike u prosjecnoj cijeni
#H1: postoji znacajna statisticka razlika u prosjecnoj cijeni
# Izdvajanje podataka za Donji i Gornji Grad
donji_grad = df[df['Lokacija'] == 'Donji Grad']['Cijena']
gornji_grad = df[df['Lokacija'] == 'Gornji Grad']['Cijena']

# Izračun prosječnih vrijednosti
mean_donji_grad = donji_grad.mean()
mean_gornji_grad = gornji_grad.mean()

# Izračun neuparenog t-testa
t, p = ttest_ind(donji_grad, gornji_grad, equal_var=True) #pretpostavimo da su im varijance jednake

# Ispis rezultata
print('Prosječna cijena najma u Donjem Gradu:', mean_donji_grad)
print('Prosječna cijena najma u Gornjem Gradu:', mean_gornji_grad)
print('t-vrijednost:', t)
print('p-vrijednost:', p) #veća od 0.05 i možemo zaključiti da ne postoje statističke značajne razlike u cijeni najma između 2 lokacije


print()
print()
print()



# Izdvajanje cijena po lokacijama
donji_grad = df[df['Lokacija'] == 'Donji Grad']['Cijena']
gornji_grad = df[df['Lokacija'] == 'Gornji Grad']['Cijena']
trešnjevka = df[df['Lokacija'] == 'Trešnjevka']['Cijena']
podsljeme = df[df['Lokacija'] == 'Podsljeme']['Cijena']
crnomerec = df[df['Lokacija'] == 'Črnomerec']['Cijena']
pescenica = df[df['Lokacija'] == 'Peščenica']['Cijena']
maksimir = df[df['Lokacija'] == 'Maksimir']['Cijena']
trnje = df[df['Lokacija'] == 'Trnje']['Cijena']
stenjevec = df[df['Lokacija'] == 'Stenjevec']['Cijena']
donja_dubrava = df[df['Lokacija'] == 'Donja Dubrava']['Cijena']
sesvete = df[df['Lokacija'] == 'Sesvete']['Cijena']
gornja_dubrava = df[df['Lokacija'] == 'Gornja Dubrava']['Cijena']
novi_zagreb = df[df['Lokacija'] == 'Novi Zagreb']['Cijena']

# provođenje ANOVA testa
statistika, p_vrijednost = stats.f_oneway(donji_grad, gornji_grad, trešnjevka, podsljeme, crnomerec, pescenica, maksimir, trnje, stenjevec, donja_dubrava, sesvete, gornja_dubrava, novi_zagreb)

# ispis rezultata
print('F-statistika =', statistika)
print('p_vrijednost =', p_vrijednost) #vrlo mala vrijednost, odbacujemo nultu hipotezu i zakljucujemo da postoji znacajna razlika izmedu cijena najma u razlicitim lokacijama



#vizualizacija podataka
import matplotlib.pyplot as plt
import numpy as np

#1 stupčasti grafikon sa prikazom lokacija i prosječnih cijena najma stanova
kategorije = ['Donji Grad', 'Gornji Grad', 'Trešnjevka', 'Podsljeme', 'Črnomerec', 'Peščenica', 'Maksimir', 'Trnje', 'Stenjevec', 'Donja Dubrava', 'Sesvete', 'Gornja Dubrava','Novi Zagreb']
vrijednosti = [donji_grad.mean(), gornji_grad.mean(), trešnjevka.mean(), podsljeme.mean(), crnomerec.mean(), pescenica.mean(), maksimir.mean(), trnje.mean(), stenjevec.mean(), donja_dubrava.mean(), sesvete.mean(), gornja_dubrava.mean(), novi_zagreb.mean()]
# crtanje stupčastog dijagrama
plt.bar(kategorije, vrijednosti)
plt.xlabel('Lokacija')
plt.ylabel('Prosječna cijena najma')
# prikazivanje dijagrama
plt.show()


#2 stupčasti grafikon sa prikazom lokacija i prosječnih veličina nekretnina
donji_grad = df[df['Lokacija'] == 'Donji Grad']['Kvadratura']
gornji_grad = df[df['Lokacija'] == 'Gornji Grad']['Kvadratura']
trešnjevka = df[df['Lokacija'] == 'Trešnjevka']['Kvadratura']
podsljeme = df[df['Lokacija'] == 'Podsljeme']['Kvadratura']
crnomerec = df[df['Lokacija'] == 'Črnomerec']['Kvadratura']
pescenica = df[df['Lokacija'] == 'Peščenica']['Kvadratura']
maksimir = df[df['Lokacija'] == 'Maksimir']['Kvadratura']
trnje = df[df['Lokacija'] == 'Trnje']['Kvadratura']
stenjevec = df[df['Lokacija'] == 'Stenjevec']['Kvadratura']
donja_dubrava = df[df['Lokacija'] == 'Donja Dubrava']['Kvadratura']
sesvete = df[df['Lokacija'] == 'Sesvete']['Kvadratura']
gornja_dubrava = df[df['Lokacija'] == 'Gornja Dubrava']['Kvadratura']
novi_zagreb = df[df['Lokacija'] == 'Novi Zagreb']['Kvadratura']

kategorije = ['Donji Grad', 'Gornji Grad', 'Trešnjevka', 'Podsljeme', 'Črnomerec', 'Peščenica', 'Maksimir', 'Trnje', 'Stenjevec', 'Donja Dubrava', 'Sesvete', 'Gornja Dubrava','Novi Zagreb']
vrijednosti = [donji_grad.mean(), gornji_grad.mean(), trešnjevka.mean(), podsljeme.mean(), crnomerec.mean(), pescenica.mean(), maksimir.mean(), trnje.mean(), stenjevec.mean(), donja_dubrava.mean(), sesvete.mean(), gornja_dubrava.mean(), novi_zagreb.mean()]
# crtanje stupčastog dijagrama
plt.bar(kategorije, vrijednosti)

plt.xlabel('Lokacija')
plt.ylabel('Prosječna kvadratura')
# prikazivanje dijagrama
plt.show()


#3 kružni grafikon sa postotkom udjela stanova po kvartovima
sizes = [len(donji_grad), len(gornji_grad), len(trešnjevka), len(podsljeme), len(crnomerec), len(pescenica), len(maksimir), len(trnje), len(stenjevec), len(donja_dubrava), len(sesvete), len(gornja_dubrava), len(novi_zagreb)]
# boje za kategorije
colors = ['blue', 'green', 'red', 'orange', 'purple', 'yellow', 'pink', 'cyan', 'magenta', 'brown', 'gray', 'teal', 'olive']
# nazivi kategorija
labels = ['Donji Grad', 'Gornji Grad', 'Trešnjevka', 'Podsljeme', 'Črnomerec', 'Peščenica', 'Maksimir', 'Trnje', 'Stenjevec', 'Donja Dubrava', 'Sesvete', 'Gornja Dubrava','Novi Zagreb']
# crtanje kružnog grafikona
plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%')
# prikazivanje dijagrama
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Histogram cijena
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Cijena', bins=10)
plt.xlabel('Cijena')
plt.ylabel('Broj stanova')
plt.title('Histogram cijena stanova u Zagrebu')
plt.show()

#Bar plot broj stanova po kvartovima
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Lokacija')
plt.xlabel('Kvartovi')
plt.ylabel('Broj stanova')
plt.title('Broj stanova po kvartovima u Zagrebu')
plt.xticks(rotation=45)
plt.show()

#prijenos dataframea u excel file
df.to_excel('nekretnine.xlsx', index=False)


# Scatter plot cijene i kvadrature stanova
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Kvadratura', y='Cijena')
plt.xlabel('Kvadratura')
plt.ylabel('Cijena')
plt.title('Scatter plot: Odnos cijene i kvadrature stanova')
plt.show()