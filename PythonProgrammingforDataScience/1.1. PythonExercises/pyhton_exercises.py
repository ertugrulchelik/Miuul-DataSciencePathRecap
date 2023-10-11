### Gorev1: Verilen degerlerin veri yapilarini inceleyiniz ###
x = 8
type(x)

y = 3.2
type(y)

z = 8j + 18
type(z)

a = "Hello World"
type(a)

b = True
type(b)

c = 23 < 22
type(c)

l = [1, 2, 3, 4]
type(l)

d = {"Name": "Jake",
     "Age": 27,
     "Address": "Downtown"}
type(d)

t = ("Machine Learning", "Data Science")
type(t)

s = {"Python", "Machine Learning", "Data Science"}
type(s)

### Görev2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.

text = "The goal is to turn data into information, and information into inside"

text.upper().replace(",", " ").replace(".", " ").split()


### Görev3: Verilen listeye aşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

# Adım 1: Verilen listenin eleman sayısına bakınız.
len(lst)

#Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.

lst[0], lst[10]

#Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
lst[:4]

#Adım 4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)


#Adım 5: Yeni bir eleman ekleyiniz.
lst.append("E")
lst

#Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")
lst

#Görev4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
dict = {'Christan': ["America", 18],
        'Daisy': ["England", 12],
        'Antonio': ["Spain", 22],
        'Dante': ["Italy", 25]}

#Adım 1: Key değerlerine erişiniz.
dict.keys()

#Adım 2: Value'lara erişiniz.
dict.values()

#Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dict["Daisy"][1] = [13]
dict

dict["Daisy"] = ["England", 13]
dict

#Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dict.update({"Ahmet": ["Turkey", 24]})
dict

#Adım 5: Antonio'yu dictionary'den siliniz.
dict.pop("Antonio")

#Görev 5: Argüman olarak bir liste alan,
# listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.

l = [2, 13, 18, 93, 22]


def func(list):
    even_list = []
    odd_list = []

    for i in range(len(list)):
        if i % 2 == 0:
            odd_list.append(i)
        else:
            even_list.append(i)

    return even_list, odd_list


func(l)

#Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
#Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken
#son üç öğrenci de tıp fakültesi öğrenci sırasına aittir.
#Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.

ogrenciler = ["Ali", "Veli", "Ayse", "Talat", "Zeynep", "Ece"]

for index, ogrenci in enumerate(ogrenciler, 1):
    if index < 4:
        print("Muh. Fak.", index, ". ogrenci", ogrenci)

    else:
        index -= 3
        print("Tip. Fak.", index, ". ogrenci", ogrenci)

#########
# Bir baska cozum
##########

for index, ogrenci in enumerate(ogrenciler):
    if index < 3:
        index = index + 1
        print("Muh. Fak.", index, ". ogrenci:", ogrenci)

    else:
        index = index - 2
        print("Tip Fak.", index, ". ogrenci:", ogrenci)

# Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer almaktadır.
# Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMP1005", "PSY1001", "HUK105", "SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

for ders_kodu, kredi, kontenjan in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {kredi} olan {ders_kodu} kodlu dersin kontenjanı {kontenjan} kişidir.")
