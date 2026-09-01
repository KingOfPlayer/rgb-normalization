# CosineSimilarity - DOCUMENTATION

## 1. Genel Bakış

### Paketin amacı ve ne yaptığı

CosineSimilarity paketi, yapay zeka modellerinden (örneğin EmbeddingExtraction düğümlerinden) elde edilen iki farklı embedding (vektör) arasındaki anlamsal benzerliği matematiksel olarak hesaplayan bir uygulamadır. Bu paket:

- İki farklı 1 boyutlu float dizisini (embedding) girdi olarak kabul eder.
- NumPy kullanarak iki vektör arasındaki Kosinüs Benzerliğini (Cosine Similarity) hesaplar.
- Algoritma çıktılarında, -1.0 ile 1.0 arasında değişen bir `similarity` (benzerlik) skoru sağlar. (1.0 mükemmel eşleşmeyi, 0.0 ilişkisizliği, -1.0 zıtlığı ifade eder).

### Temel özellikler

- ✅ İki vektör arasında yüksek performanslı kosinüs benzerliği hesaplama
- ✅ Pydantic tabanlı model tanımları (girdi/çıktı/konfigürasyon)
- ✅ Sıfıra bölme (DivisionByZero) hatalarına karşı korumalı (Zero-vector kontrolü)
- ✅ NumPy kullanılarak optimize edilmiş matris/vektör işlemleri
- ✅ NovaVision platformu Update/Build süreçleri için `try-except` tabanlı kurşun geçirmez import yönetimi

### Desteklenen sınıflar / modeller / tipler
| ID | İsim | Açıklama |
|----|------|---------|
| 1  | `CosineSimilarity` | Capsulenin ana executor sınıfı — matematiksel benzerlik hesaplamasını yapar |
| 2  | `PackageModel` | Paket genel yapı tanımı (configs, executor vb.) |
| 3  | `InputEmbedding1` | Pydantic input modeli — birinci embedding vektörü (List[float]) |
| 4  | `InputEmbedding2` | Pydantic input modeli — ikinci embedding vektörü (List[float]) |
| 5  | `OutputSimilarity` | Pydantic output modeli — -1.0 ile 1.0 arası float similarity skoru |

---

## 2. Mimari ve Teknolojiler

### Teknoloji Stack'i
- Framework: Python 3.x
- İşleme ve Matematik: NumPy
- Veri Doğrulama ve Modelleme: Pydantic
- SDK Bileşenleri: `sdks.novavision` (Component, Executor, PackageHelper vb.)

### Her teknolojinin rolü ve kullanımı (kart formatında)

- Python 3.x
  - Rol: Ana programlama dili
  - Kullanım: Paket mantığı, Pydantic modeller, iş mantığı

- NumPy
  - Rol: Yüksek performanslı matematiksel hesaplama
  - Kullanım: Listeleri `ndarray` formatına çevirmek, vektörlerin normlarını (`np.linalg.norm`) ve nokta çarpımlarını (`np.dot`) hesaplamak.

- Pydantic
  - Rol: Input/Output/Config model validasyonu ve schema tanımı
  - Kullanım: `PackageModel.py` içerisindeki modellerin doğrulanması (örn: vektörlerin en az 1 elemanlı olması, sonucun -1 ile 1 arasında sınırlandırılması).

- sdks.novavision (SDK Bileşenleri)
  - Rol: Paket geliştirme için yardımcı sınıflar
  - Kullanım: `Component` sınıfı ile executor çalışma mantığı, Request/Response modelleri üzerinden sistem iletişimi.

### Proje yapısı (tree formatında)

```
cosine-similarity/
├── README.md                            # Kısa proje açıklaması ve kullanım kılavuzu
├── DOCUMENTATION.md                     # (Bu dosya) Detaylı dokümantasyon
├── setup.py                             # Paket kurulumu
├── src/
│   ├── __init__.py
│   ├── executors/
│   │   ├── __init__.py
│   │   └── CosineSimilarity.py          # Executor sınıfı: main logic ve matematiksel hesaplama
│   ├── models/
│   │   ├── __init__.py
│   │   └── PackageModel.py              # Pydantic modeller: Inputs, Outputs, Configs, Request, Response
│   └── utils/
│       └── response.py                  # Response oluşturma helper'ı (Güvenlik ağlı import içerir)
```

Açıklamalar:
- `CosineSimilarity.py` — Kosinüs benzerliğini uygulayan `Component` executor'ı içerir. `run()` metodu, girdileri alıp Numpy ile hesaplar ve `build_response()` ile sonucu paketler. Ayrıca NovaVision update sistemindeki klasör adı değişikliklerine karşı try-except korumalı import'a sahiptir.
- `PackageModel.py` — Pydantic modeller: Input (`embedding_1`, `embedding_2`), Output (`similarity`) için sınır ve tip yapılandırmaları.
- `utils/response.py` — executor context'inden çıkış modelini (OutputSimilarity) oluşturma.

---

## 3. Executor'lar ve Çalışma Modları

### `CosineSimilarity` (Tam path: `src/executors/CosineSimilarity.py`)

- Amaç:
  - Verilen iki embedding (vektör) listesini alarak, bunların yönsel açıdan birbirlerine ne kadar benzer olduğunu gösteren kosinüs benzerliği skorunu hesaplamak.

- Kullanım senaryosu:
  - ✅ Metin ve Görüntü (Image & Text) eşleştirme ve arama.
  - ✅ Yüz tanıma veya nesne benzerliği algoritmaları.
  - ✅ Tavsiye sistemlerinde anlamsal mesafe ölçümü.

- İşleyiş (numaralı adımlar):
  1. `run()` çağrısında request üzerinden `embedding_1` ve `embedding_2` listeleri çekilir.
  2. `calculate_similarity` metodu çağrılarak bu listeler `np.array` yapısına dönüştürülür.
  3. İki vektörün boyutlarının (dimensionality) aynı olup olmadığı kontrol edilir, farklıysa hata fırlatılır.
  4. Vektörlerin büyüklüğü (`norm`) `np.linalg.norm` ile hesaplanır. Herhangi biri `0` ise sıfıra bölme hatasını önlemek için `0.0` dönülür.
  5. İki vektörün nokta çarpımı (dot product), normların çarpımına bölünür.
  6. Sonuç `np.clip` ile -1.0 ve 1.0 aralığına zorlanarak geri döndürülür.
  7. `build_response` ile Pydantic `Response` nesnesi oluşturulur.

- Python sınıfı (tam path): `components.CosineSimilarity.src.executors.CosineSimilarity.CosineSimilarity`

- Temel metodlar:
  - `__init__(self, request, bootstrap)` : Executor başlatma, request -> PackageModel mapping, input verilerinin alınması.
  - `calculate_similarity(self, emb1, emb2)` : Numpy ile Kosinüs benzerliği hesaplayan temel matematiksel fonksiyon.
  - `run(self)` : End-to-end işlem; parametreleri çekme, similarity hesaplama, response oluşturma.

---

## 4. Girdi (Input) Parametreleri

### 4.1 `InputEmbedding1` ve `InputEmbedding2` (Pydantic Model)
```python
class InputEmbedding1(Input):
    name: Literal["embedding_1"] = "embedding_1"
    value: List[float] = Field(..., description="Embedding vector 1", min_items=1)
    type: Literal["list"] = "list"
```

- Tanım: Yapay zeka modellerinden (Clip vb.) çıkan float tabanlı öznitelik vektörleri. Her iki input da aynı yapıya (List[float]) sahiptir.
- Özellikler:
  - `name`: "embedding_1" veya "embedding_2"
  - `value`: En az 1 elemanı olan float dizisi (`List[float]`)
  - `type`: `list`

- Obje yapısı örneği (JSON):
```json
{
  "name": "embedding_1",
  "value": [0.123, -0.456, 0.789, 0.012],
  "type": "list"
}
```
- Kullanıldığı executor'lar:
  - CosineSimilarity: ✅

---

## 5. Konfigürasyon (Config) Parametreleri

Bu paket tamamen "Plug & Play" mantığında çalıştığı ve dışarıdan bir ayar veya parametre almadığı için kullanıcı tarafından arayüz üzerinden girilecek bir konfigürasyon (Config) parametresi **bulunmamaktadır**. Tüm veri sadece Input kablolarından (cables) gelir.

---

## 6. Çıktı (Output) Parametreleri

### 6.1 `OutputSimilarity` (Pydantic Model)
```python
class OutputSimilarity(Output):
    name: Literal["similarity"] = "similarity"
    value: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score")
    type: Literal["number"] = "number"
```
- Tanım: Hesaplanan matematiksel benzerlik skoru. Değer aralığı Pydantic tarafından matematiksel gerçekliğe uygun şekilde `-1.0` ile `1.0` arasına sınırlandırılmıştır (`ge=-1.0, le=1.0`).

- Yapı örneği (JSON):
```json
{
  "name": "similarity",
  "type": "number",
  "value": 0.26236629486083984
}
```
- Kullanıldığı executor'lar:
  - CosineSimilarity: ✅

---

## 7. Veri Modelleri

### PackageModel hiyerarşisi (ASCII tree)
```
PackageModel (Package)
├── configs (PackageConfigs)
│   └── executor (ConfigExecutor)
│       └── value (CosineSimilarityExecutor)
│           └── value (CosineSimilarityRequest or Response)
│               ├── inputs (CosineSimilarityInputs) -> embedding_1, embedding_2
│               └── outputs (CosineSimilarityOutputs) -> similarity
```

### Request / Response akışları (her executor için)

- CosineSimilarity (ASCII sequence):
```
[Client] ----JSON Request containing-> [PackageModel (configs->executor->value->CosineSimilarityRequest)]
      |
      V
[Executor: CosineSimilarity] --run()
      |
      V
1. Extract embedding_1 and embedding_2 from request
2. calculate_similarity(emb1, emb2)
3. numpy conversion, dimension check, norm calculation, dot product
4. Build OutputSimilarity object
5. build_response(context) -> PackageHelper -> PackageModel Response JSON
      |
      V
[Client] <- JSON Response (CosineSimilarityResponse with similarity output)
```

---

## 8. Metodoloji ve Algoritmalar

### 8.1 Kosinüs Benzerliği Hesaplaması (Cosine Similarity)
- Amaç: Çok boyutlu uzayda, iki vektör arasındaki açının kosinüsünü ölçerek, bu vektörlerin yönsel olarak birbirine ne kadar yakın olduğunu bulmak.

- Adımlar (numaralı):
  1. Gelen `List[float]` verilerini `numpy.ndarray` formatına çevir (`np.float32`).
  2. İki vektörün boyutlarının (eleman sayılarının) birbirine eşit olduğunu doğrula.
  3. Birinci ve ikinci vektörün büyüklüklerini (`norm`) `np.linalg.norm` ile hesapla.
  4. Eğer normlardan herhangi biri `0` ise, tanımsızlık (`DivisionByZero`) oluşturmamak için anında `0.0` dön.
  5. İki vektörün iç çarpımını (`dot product`) hesapla.
  6. İç çarpımı, iki normun çarpımına böl.
  7. Çıkan sonucu virgülden sonra kayan nokta hatalarına karşı `np.clip` ile `-1.0` ile `1.0` arasında sınırla ve geri dön.

- Pseudo-code:
```python
def calculate_similarity(emb1, emb2):
    vec1 = np.array(emb1, dtype=np.float32)
    vec2 = np.array(emb2, dtype=np.float32)

    if vec1.shape != vec2.shape:
        raise ValueError("Boyutlar esit olmalidir.")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(np.clip(sim, -1.0, 1.0))
```

- Avantajlar:
  - ✅ **Hız:** Numpy kullanılarak C düzeyinde matris çarpımı yapıldığı için devasa vektörlerde (512, 1024 boyutlu) bile hesaplama milisaniyeler sürer.
  - ✅ **Güvenlik:** Sıfır vektörü kontrolleri sayesinde uç durumlarda API'nin çökmesi engellenir.
  - ✅ **Standartlık:** Çıktı tamamen matematiksel standartlara uygundur ve anomali barındırmaz.
