##################################################
# List Comprehensions
##################################################

# ###############################################
# # GÖREV 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini
# büyük harfe çeviriniz ve başına NUM ekleyiniz.
# ###############################################
#

import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = sns.load_dataset("car_crashes")
df.head()
df.columns
df.info()


num_cols = [col for col in df.columns if df[col].dtype != "O"]

["NUM_" + col.upper() if col in num_cols else col.upper() for col in df.columns]


# ###############################################
# # GÖREV 2: List Comprehension yapısı kullanarak car_crashes verisindeki
# isminde "no" barındırmayan değişkenlerin isimlerininin sonuna "FLAG" yazınız.
# ###############################################
#

df.columns

[col for col in df.columns if "no" not in col]

[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]



# ###############################################
# # Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden
# FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.
# ###############################################

og_list = ["abbrev", "no_previous"]

# # Notlar:
# # Önce yukarıdaki listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
# # Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz adını new_df olarak isimlendiriniz.
#
# # Beklenen çıktı:
#
# #    total  speeding  alcohol  not_distracted  ins_premium  ins_losses
# # 0 18.800     7.332    5.640          18.048      784.550     145.080
# # 1 18.100     7.421    4.525          16.290     1053.480     133.930
# # 2 18.600     6.510    5.208          15.624      899.470     110.350
# # 3 22.400     4.032    5.824          21.056      827.340     142.390
# # 4 12.000     4.200    3.360          10.920      878.410     165.630
#

new_cols = [col for col in df.columns if col not in og_list]

new_df = df[new_cols]

new_df.head()

##################################################
# Pandas Alıştırmalar
##################################################

import numpy as np
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#########################################
# Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
#########################################

df = sns.load_dataset("titanic")
df.head()
df.shape

#########################################
# Görev 2: Yukarıda tanımlanan Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
#########################################

df["sex"].value_counts()

#########################################
# Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
#########################################

df.nunique()

#########################################
# Görev 4: pclass değişkeninin unique değerleri bulunuz.
#########################################

df["pclass"].unique()

df["pclass"].nunique()

#########################################
# Görev 5:  pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
#########################################

df["pclass"].nunique()

df["parch"].nunique()

#veya baska bir sekil

df[["pclass", "parch"]].nunique()


#########################################
# Görev 6: embarked değişkeninin tipini kontrol ediniz.
# Tipini category olarak değiştiriniz. Tekrar tipini kontrol ediniz.
#########################################

df["embarked"].dtype

df["embarked"] = df["embarked"].astype("category")

df["embarked"].dtype


#########################################
# Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] == "C"].count()

df[df["embarked"] == "C"].info

df[df["embarked"] == "C"].head()

#########################################
# Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
#########################################

df[df["embarked"] != "S"].info

#########################################
# Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
#########################################

df[(df["age"] < 30) & (df["sex"] == "female")].head()

df[(df["age"] < 30) & (df["sex"] == "female")].info

#########################################
# Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.
#########################################

df[(df["fare"] > 500) | (df["age"] > 70)]

df[(df["fare"] > 500) | (df["age"] > 70)].info

#########################################
# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
#########################################

df.isnull().sum()

#########################################
# Görev 12: who değişkenini dataframe'den düşürün.
#########################################

df.head()

df.drop("who", axis=1, inplace=True)
df.head()

#########################################
# Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
#########################################

mode_deck = df["deck"].mode()[0]

df["deck"].isnull().sum()

df["deck"].fillna(mode_deck, inplace=True)

df["deck"].isnull().sum()

#########################################
# Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurun.
#########################################

median_age = df["age"].median()

df["age"].isnull().sum()

df["age"].fillna(median_age, inplace=True)

df["age"].isnull().sum()

#########################################
# Görev 15: survived değişkeninin Pclass ve Cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
#########################################

df.groupby(["pclass", "sex"]).agg({"survived": ["sum","count","mean"]})

#########################################
# Görev 16:  30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazınız.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında yeni bir değişken oluşturunuz. (apply ve lambda yapılarını kullanınız)
#########################################


def age_30(age):
    if age < 30:
        return 1
    else:
        return 0


df["age_flag"] = df["age"].apply(lambda x: age_30(x))


# fonksiyon yazmadan da bu islemi yapabilirdik

df["age_flag1"] = df["age"].apply(lambda x: 1 if x<30 else 0)

df.head()

#########################################
# Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
#########################################

df = sns.load_dataset("tips")
df.head()
df.shape
df.info()

#########################################
# Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill  değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

#########################################
# Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.head()

df.groupby(["day", "time"]).agg({"total_bill": ["sum", "min", "max", "mean"]})

#########################################
# Görev 20:Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.
#########################################

df.head()

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"],
                                                                         "tip": ["sum", "min", "max", "mean"]})

#########################################
# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir?
#########################################\

df.head()

df[(df["size"] < 3) & (df["total_bill"] > 10)].agg({"total_bill": "mean"})

# loc yapisi kullanilarak da secim islemi yapilabilir
df.loc[(df["size"] < 3) & (df["total_bill"] >10 ) , "total_bill"].mean()

#########################################
# Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturun. Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
#########################################

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

#########################################
# Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.
#########################################

new_data_frame = df.sort_values("total_bill_tip_sum", ascending=False)[:30]

# degiskenin tipini kontrol ettigimizde dataframe seklinde goruyoruz
type(new_data_frame)