# Letterboxd Film Türü Tahmini

Film açıklamalarından beş tür etiketini tahmin eden çok etiketli metin sınıflandırma projesi. Her film için **Action, Comedy, Drama, Horror** ve **Romance** etiketlerinin her biri ayrı ayrı tahmin edilir; dolayısıyla bir filme birden fazla tür atanabilir.

Bu depo, BIL/YAP 470 boyunca U2–U8 haftalarında aynı Letterboxd projesine uygulanan yöntemlerin kaynak kodunu, eğitilmiş modellerini, ön işleme dosyalarını, örnek girdi dosyasını ve karşılaştırma raporunu içerir. Ham veri dosyaları depoya eklenmemiştir.

## Proje özeti

- **Girdi:** Film açıklaması metni
- **Çıktı:** Beş tür için olasılık ve 0/1 tahmini
- **Veri kaynağı:** [Letterboxd Movies Dataset, Kaggle](https://www.kaggle.com/datasets/gsimonx37/letterboxd), proje sırasında kullanılan sürüm 2
- **Örneklem:** 10.000 film; aynı beş etiket bir filmde birlikte bulunabilir
- **Sabit bölme:** 7.020 eğitim, 1.490 doğrulama ve 1.490 test filmi
- **Rastgelelik tohumu:** `470`
- **Birincil metrik:** Macro-F1. Her etikete eşit ağırlık verdiği için az görülen türlerin başarısı da değerlendirmeye katılır.

| Etiket | Pozitif film | Örneklem oranı |
|---|---:|---:|
| Action | 1.071 | %10,71 |
| Comedy | 3.225 | %32,25 |
| Drama | 5.730 | %57,30 |
| Horror | 1.428 | %14,28 |
| Romance | 1.326 | %13,26 |

## Haftalık yöntemler

| Hafta | Yöntem | Girdi temsili ve seçilen ayar |
|---|---|---|
| U2 | Uygunluk ağırlıklı k-en yakın komşu (kNN), beş etiket için çok çıktılı tahmin | 1.000 TF-IDF özellik, 704 bileşenli PCA, `k=201`, uzaklığa göre ağırlık |
| U3 | One-vs-rest lojistik regresyon | 1.000 TF-IDF özellik, 704 bileşenli PCA, L2, `lambda=0.0001` |
| U4 | One-vs-rest karar ağacı | 1.000 TF-IDF özellik, 704 bileşenli PCA, entropi, `max_depth=5` |
| U5 | Rastgele orman | 1.000 TF-IDF özellik, 704 bileşenli PCA, 200 ağaç, `max_features=sqrt` |
| U5 | Histogram gradyan artırma | 1.000 TF-IDF özellik, 704 bileşenli PCA, öğrenme oranı `0.1`, 200 iterasyon |
| U6 | One-vs-rest doğrusal SVM | 1.000 TF-IDF özelliği, `C=0.1`, dengeli sınıf ağırlıkları |
| U7 | PyTorch çok katmanlı algılayıcı (MLP) | 1.000 TF-IDF özelliği, gizli katmanlar `[128, 64]`, dropout `0.2`, öğrenme oranı `0.001` |
| U8 | MLP ayar karşılaştırması | Dört mimari/dropout/öğrenme oranı ayarı; doğrulama macro-F1 ile seçim |

U8 ödevindeki CNN görüntü girdisi içindir. Bu proje metin kullandığından proje verisi üzerinde MLP ayarları karşılaştırılmış, CNN uygulanabilir yöntem olarak kullanılmamıştır.

## Sonuçları okuma

`REPORT.md`, iki ayrı değerlendirme kümesini birbirine karıştırmadan raporlar:

1. **Haftalık ödevlerin ortak test bölümü:** Haftalık çalışmalarda kaydedilen Macro-F1 sonuçları aşağıdaki tabloda yer alır.
2. **Bu depodaki yeniden eğitilmiş model dosyalarının doğrulama bölümü:** Her kaydedilmiş modelin doğrulama Macro-F1, Macro-AUC ve eşiği `artifacts/model_registry.json` içinde bulunur. Bunlar test sonuçları değildir.

| Hafta / yöntem | Haftalık ortak test Macro-F1 |
|---|---:|
| U2 kNN | 0.4726 |
| U3 lojistik regresyon | **0.5381** |
| U4 karar ağacı | 0.3939 |
| U5 rastgele orman | 0.4093 |
| U5 gradyan artırma | 0.4663 |
| U6 doğrusal SVM | 0.5092 |
| U7 MLP | 0.4996* |

Tüm etiketleri negatif tahmin etmek Macro-F1 `0.0000` verir. Yalnızca en sık tür olan Drama'yı tahmin eden basit taban çizgisinin Macro-F1 değeri `0.1463` olmuştur. Lojistik regresyonun `0.5381` test Macro-F1 değeri bu taban çizgisinden **0.3918 puan yüksektir** (yaklaşık 3.68 katı). Bu karşılaştırma açıklamalarda türlere ilişkin öğrenilebilir sinyal bulunduğunu gösterir; başka veri kümelerinde aynı sonucun alınacağını garanti etmez.

`*` U7 için haftalık formda doğrulama Macro-F1 `0.5125` olarak kaydedilmişti. Tablodaki `0.4996`, haftalık kayıtların dışında, ortak test bölümü üzerinde yapılan ek değerlendirmedir. U8 ayar aramasında doğrulama verisinde en iyi bulunan ayar `hidden=[128, 64]`, `dropout=0.2`, öğrenme oranı `0.0005` olmuştur; U8 için burada haftalık ortak test sonucu iddia edilmemektedir.

![Haftalık yöntemlerin ortak test Macro-F1 karşılaştırması](figures/test_macro_f1.png)

Sonuçlar yöntemlerin doğrudan ve tek başına karşılaştırması değildir: U2–U5 PCA ile daraltılmış özellikler kullanırken U6–U7 ham TF-IDF özellikleri kullanır. Tekli kelime TF-IDF sözcük sırasını ve bağlamı yakalayamaz; ayrıca türler dengesizdir ve bir film birden fazla türe ait olabilir. Sonuçlar tek sabit bölmeye dayanır.

## Kurulum

Python 3.10 veya üstü gerekir. Depo klasöründe aşağıdaki komutları çalıştırın:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Scikit-learn model dosyaları scikit-learn `1.6.1` ile kaydedildi. Bağımlılık dosyası bu sürümü sabitler. Modelleri yüklerken depo içindeki `artifacts/` klasörünün de bulunduğundan emin olun.

## Hazır modelle tahmin

Örnek girdi `examples/sample_descriptions.csv` dosyasında bulunur. Aynı biçimde kendi CSV dosyanızı oluşturun: `description` sütunu zorunludur; varsa `id` sütunu çıktı dosyasına aktarılır.

```csv
id,description
film-001,"A detective investigates a series of mysterious events in a small town."
film-002,"Two friends travel across the country and discover an unexpected romance."
```

Örnek olarak lojistik regresyonla tahmin üretin:

```bash
python -m letterboxd_project.predict examples/sample_descriptions.csv --model u3_logistic_regression --artifacts artifacts --output predictions.csv
```

`predictions.csv` dosyasında her film için beş olasılık (`Action_prob` gibi) ve beş ikili karar (`Action_pred` gibi) bulunur. Olasılıklar 0–1 arasındadır. Birden fazla `*_pred` değeri 1 olabilir. Tahminler her model için doğrulama verisinde seçilmiş eşik kullanılarak hesaplanır; varsayılan eşik her modelde ve etikette otomatik olarak `0.5` değildir.

`artifacts/model_registry.json` içindeki model anahtarlarından birini `--model` ile seçebilirsiniz:

```text
u2_knn
u3_logistic_regression
u4_decision_tree
u5_random_forest
u5_gradient_boosting
u6_linear_svm
u7_mlp
u8_mlp_dropout0
u8_mlp_dropout02
u8_mlp_large
u8_mlp_lr0005
u8_selected_mlp
```

`u8_selected_mlp`, doğrulama sonuçlarında en yüksek macro-F1 değerine sahip U8 modeline giden kısayoldur. Her modelin eşik ve doğrulama ölçümleri kayıt dosyasında yer alır. SVM için döndürülen değer kalibre edilmiş olasılık değildir; karar skoru sigmoid dönüşümüyle 0–1 aralığına taşınır.

### Python içinden kullanma

```python
from letterboxd_project.predict import predict

descriptions = [
    "A detective investigates a series of mysterious events in a small town.",
    "Two friends travel across the country and discover an unexpected romance.",
]

predictions, probabilities = predict(
    descriptions,
    model_name="u3_logistic_regression",
    artifact_dir="artifacts",
)

print(predictions)   # Her etiket için *_prob ve *_pred sütunları
print(probabilities) # Sıra: Action, Comedy, Drama, Horror, Romance
```

## Modelleri yeniden eğitme

Ham veri depoda bulunmaz. [Kaggle veri kaynağından](https://www.kaggle.com/datasets/gsimonx37/letterboxd) `movies.csv` ve `genres.csv` dosyalarını indirin ve aşağıdaki konumlara koyun:

```text
data/raw/movies.csv
data/raw/genres.csv
```

Bu depodaki `data/splits/letterboxd_split_u5.csv` dosyası, 10.000 film için sabit eğitim/doğrulama/test ID bölmesini içerir. Aynı örneklem ve bölmeyi yeniden oluşturmak için CSV'lerdeki ID'lerin ve açıklamaların kullanılan veri sürümüyle eşleşmesi gerekir. Ardından tüm yöntemleri ve U8'in dört MLP ayarını yeniden eğitin:

```bash
python -m letterboxd_project.train_all --data-dir data/raw --split-manifest data/splits/letterboxd_split_u5.csv --output artifacts --with-u8-search
```

Eğitim betiği `SEED=470` kullanır. TF-IDF vektörleştiricisi ve PCA yalnızca eğitim bölümüne fit edilir. Model ve ortak karar eşiği doğrulama bölümüyle seçilir. Test bölümü bu betikte fit, eşik seçimi ya da raporlanan değerlendirme için kullanılmaz. Eğitim tamamlandığında modeller, ön işleme nesneleri ve doğrulama kayıtları `artifacts/` içine yazılır. Bu komut mevcut `artifacts/` dosyalarının üzerine yazabilir; özgün modelleri korumak için `--output` değerini farklı bir klasör yapın.

## Depo içeriği

```text
artifacts/       Eğitilmiş modeller, TF-IDF/PCA nesneleri ve model kayıt dosyası
data/splits/     Sabit örnek ID ve bölme listesi; ham film metinleri içermez
examples/        Deneme için küçük örnek girdi CSV'si
figures/         Rapordaki karşılaştırma grafiği
src/             Tahmin ve yeniden eğitim Python kodu
REPORT.md        Veri, metrik, kıyaslama, taban çizgisi ve sınırlamalar raporu
```

## Veri ve lisans

Veri kaynağı proje sırasında [Simon Garanin'in Kaggle Letterboxd veri kümesi](https://www.kaggle.com/datasets/gsimonx37/letterboxd) (sürüm 2) olarak kaydedildi. Ders projesi kaydında lisans GPL-3 olarak belirtilmişti. Kaynak sayfadaki güncel lisans ve kullanım koşullarını kontrol edin. Ham CSV dosyaları bu depoda dağıtılmaz.

Bu depodaki kaynak kod ve eğitilmiş ağırlıklar için ayrıca bir açık kaynak lisansı seçilmemiştir. Bir lisans dosyası eklenene kadar başkalarına yeniden kullanım izni verildiği varsayılmamalıdır. Veri kümesinin lisansı ile bu depodaki kodun lisansı birbirinden ayrıdır.
