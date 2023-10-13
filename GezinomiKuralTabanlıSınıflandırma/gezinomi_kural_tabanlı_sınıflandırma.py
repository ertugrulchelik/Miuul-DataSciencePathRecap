#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Gezinomi yaptığı satışların bazı özelliklerini kullanarak seviye tabanlı (level based) yeni satış tanımları
# oluşturmak ve bu yeni satış tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin
# şirkete ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.
# Örneğin: Antalya’dan Herşey Dahil bir otele yoğun bir dönemde gitmek isteyen bir müşterinin
# ortalama ne kadar kazandırabileceği belirlenmek isteniyor.
#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################
# Soru 1: miuul_gezinomi.xlsx dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = pd.read_excel('/Users/ertugrulcelik/Desktop/DataSciencePathRecap/1.3. GezinomiKuralTabanlıSınıflandırma/miuul_gezinomi.xlsx')

df.head()
df.shape
df.info()
df.describe().T
df.isnull().values.any()
df.isnull().any()
df.isnull().sum()

# Soru 2: Kaç unique şehir vardır? Frekansları nedir?

df["SaleCityName"].nunique()

df["SaleCityName"].value_counts()

# Soru 3: Kaç unique Concept vardır?

df["ConceptName"].nunique()

# Soru 4: Hangi Concept'dan kaçar tane satış gerçekleşmiş?k

df["ConceptName"].value_counts()

df.groupby("ConceptName").agg({"Price": "count"})

# Soru 5: Şehirlere göre satışlardan toplam ne kadar kazanılmış?

df.groupby("SaleCityName").agg({"Price": "sum"})

# Soru 6: Concept türlerine göre göre ne kadar kazanılmış?

df.groupby("ConceptName").agg({"Price": "sum"})

# Soru 7: Şehirlere göre PRICE ortalamaları nedir?

df.groupby("SaleCityName").agg({"Price": "mean"})

# Soru 8: Conceptlere  göre PRICE ortalamaları nedir?

df.groupby("ConceptName").agg({"Price": "mean"})

# Soru 9: Şehir-Concept kırılımında PRICE ortalamaları nedir?

df.groupby(["SaleCityName", "ConceptName"]).agg({"Price": "mean"})


#############################################
# GÖREV 2: satis_checkin_day_diff değişkenini EB_Score adında yeni bir kategorik değişkene çeviriniz.
#############################################

#agg_df["CAT_AGE"] = pd.cut(df["AGE"], [0, 18, 25, 35, 65, 75], labels=["0_18", "19_25", "26_35", "36_45", "66_75"])
#agg_df.head(50)

df.head()
df["SaleCheckInDayDiff"].nunique()

labels = ["Last Minuters", "Potential Planners", "Planners", "Early Bookers"]

df["EB_Score"] = pd.qcut(df["SaleCheckInDayDiff"], q=4, labels=labels)

df["EB_Score"].head()
df["EB_Score"].info


#############################################
# GÖREV 3: Şehir,Concept, [EB_Score,Sezon,CInday] kırılımında ücret ortalamalarına ve frekanslarına bakınız
#############################################
# Şehir-Concept-EB Score kırılımında ücret ortalamaları

df.groupby(["SaleCityName", "ConceptName", "EB_Score"]).agg({"Price": ["mean","count"]})

# Şehir-Concept-Sezon kırılımında ücret ortalamaları

df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": ["mean","count"]})

# Şehir-Concept-CInday kırılımında ücret ortalamaları

df.groupby(["SaleCityName", "ConceptName", "CInDay"]).agg({"Price": ["mean","count"]})


#############################################
# GÖREV 4: City-Concept-Season kırılımın çıktısını PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False)
agg_df.head()

#############################################
# GÖREV 5: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()

agg_df.reset_index(inplace=True)

agg_df.head()

#############################################
# GÖREV 6: Yeni level based satışları tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# sales_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
agg_df['sales_level_based'] = agg_df[["SaleCityName", "ConceptName", "Seasons"]].agg(lambda x: '_'.join(x).upper(), axis=1)

#############################################
# GÖREV 7: Personaları segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz
# segmentleri betimleyiniz


# df["EB_Score"] = pd.qcut(df["SaleCheckInDayDiff"], q=4, labels=labels)

agg_df["SEGMENT"] = pd.qcut(agg_df["Price"], 4, labels=["D", "C", "B", "A"])

agg_df["SEGMENT"].head(100)

agg_df.groupby("SEGMENT").agg({"Price": ["mean", "max", "sum"]})

#############################################
# GÖREV 8: Oluşan son df'i price değişkenine göre sıralayınız.
# "ANTALYA_HERŞEY DAHIL_HIGH" hangi segmenttedir ve ne kadar ücret beklenmektedir?
#############################################

# agg_df = df.groupby(["SaleCityName", "ConceptName", "Seasons"]).agg({"Price": "mean"}).sort_values("Price", ascending=False)

agg_df.sort_values("Price")

new_user = "ANTALYA_HERSEY_DAHIL_HIGH"

agg_df[agg_df["sales_level_based"] == new_user]

