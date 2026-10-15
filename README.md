# Automatska klasifikacija kategorija proizvoda

**IT Akademija · Task 3 · Mašinsko učenje · Nikola Novkov**

Projekat predviđa jednu od **10 kategorija** na osnovu **naziva proizvoda**. Cilj je da se ubrza dodavanje proizvoda u katalog. `Product ID`, `Merchant ID`, popularnost, ocene prodavca i datum nisu korišćeni u modelu: novi korisnik unosi samo naslov, pa te vrednosti ne bi bile dostupne pri predikciji.

## Brzi početak

Potrebni su Python 3.10+ i internet samo za instaliranje zavisnosti:

```bash
python -m pip install -r requirements.txt
python predict_category.py
```

Primer interaktivne upotrebe:

```text
Naziv proizvoda: olympus e m10 mark iii geh use silber
Predlozena kategorija: Digital Cameras

Naziv proizvoda: smeg sbs8004po
Predlozena kategorija: Fridge Freezers

Naziv proizvoda: izlaz
Kraj programa.
```

Alternativno:

```bash
python predict_category.py --text "bosch serie 4 kgv39vl31g"
```

Model je već istreniran i nalazi se u `models/product_category_model.pkl`.

## Pokretanje treniranja i analize

```bash
python train_model.py
python -m unittest discover -s tests -v
jupyter notebook notebooks/product_classifier_analysis.ipynb
```

`train_model.py` učitava `data/products.csv`, čisti podatke, pravi stratifikovanu podelu **80:20** (`random_state=42`), trenira četiri kandidata i poredi **accuracy, macro precision, macro recall i macro F1**. Pobednik je model sa najvećim macro F1; tek posle testiranja ponovo se trenira na **svim očišćenim podacima** i čuva kao `.pkl`. TF-IDF se prilagođava samo trening porukama prilikom evaluacije, pa se sprečava curenje informacija iz test skupa.

## Čišćenje i karakteristike

- Nazivi kolona s viškom razmaka se normalizuju.
- Odbacuju se prazni naslovi i nepoznate/nepostojeće kategorije.
- `CPU → CPUs`, `Mobile Phone → Mobile Phones`, `fridge → Fridges`.
- Ponovljeni identični naslovi se svode na jedan red kako isti naziv ne bi bio u oba skupa.
- Ako isti normalizovani naslov ima različite kategorije, taj naslov je isključen kao kontradiktoran.
- TF-IDF karakteristike: reči (1–2 grama), karakteri (3–5 grama), kao i njihova kombinacija.
- U svesci se analiziraju dodatne osobine naslova: dužina, broj reči, prisustvo brojeva. Te izvedene vrednosti služe za istraživanje, a finalna predikcija koristi karakter-n-grame.

## Poređeni modeli

1. Logistic Regression + word TF-IDF
2. Multinomial Naive Bayes + word TF-IDF
3. Linear SVC + character TF-IDF
4. Linear SVC + kombinovani word/character TF-IDF

Stvarni rezultati na izdvojenom test skupu su u `reports/model_comparison.csv` i u izvršenoj Jupyter svesci. Rezultati su ponovljivi uz iste verzije biblioteka, ali mogu blago odstupati u drugim okruženjima.

**Napomena o proceni:** gotovo identični naslovi s različitim nijansama i dalje mogu postojati u oba skupa. Zbog toga 80:20 evaluacija možda precenjuje kvalitet na potpuno novim proizvodnim linijama. U produkciji bi bilo korisno testirati i na grupama proizvođača/modela koje model ranije nije video.

## Struktura

```text
product-category-classifier/
├── data/products.csv                     # originalni podaci
├── models/product_category_model.pkl    # gotov istreniran model
├── notebooks/product_classifier_analysis.ipynb
├── reports/
│   ├── model_comparison.csv
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── training_summary.json
├── src/data_utils.py                     # ciscenje podataka
├── tests/test_smoke.py
├── train_model.py
├── predict_category.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Ograničenja i nastavak razvoja

Model daje **predlog**, a ne potvrđenu kategoriju. Za proizvode s neuobičajenim nazivima preporučuje se ručna provera. Sledeće iteracije: kvalitetniji labels, analiza konfuzija između *Fridges* i *Fridge Freezers*, prag za preusmeravanje nepoznatih proizvoda čoveku, provera na potpuno novim brendovima i periodično ponovno treniranje.

**Bezbednost:** učitavajte `.pkl` samo iz pouzdanog izvora — `pickle/joblib` pri učitavanju može izvršiti proizvoljan kod.

## Kako objaviti projekat na GitHub-u

1. Prijavi se na [GitHub](https://github.com/) i izaberi **New repository**.
2. Ime: `product-category-classifier`, vidljivost: **Public**. Ne pravi dodatni README prilikom otvaranja repozitorijuma.
3. U repozitorijumu klikni **uploading an existing file** / **Add file → Upload files**. Raspakuj preuzetu ZIP arhivu na računaru i prevuci sve fajlove i foldere iz projektne fascikle u GitHub upload (zadržati strukturu direktorijuma). Alternativa je `git` komanda ispod.
4. Klikni **Commit changes** i osveži početnu stranicu repozitorijuma.
5. Proveri da repozitorijum sadrži `data/products.csv`, `models/product_category_model.pkl`, `.ipynb`, obe `.py` skripte i README, pa kopiraj URL i pošalji ga nastavniku.

Ako koristiš Git instaliran na računaru, u korenu raspakovanog projekta:

```bash
git init
git add .
git commit -m "Add complete product classification ML project"
git branch -M main
git remote add origin https://github.com/TVOJ_USERNAME/product-category-classifier.git
git push -u origin main
```

U linku za `git remote` zameni `TVOJ_USERNAME` svojim korisničkim imenom. Koristi GitHub autentifikaciju; **nemoj upisivati lozinku/token u kod, notebook ili repozitorijum**.

## Odluke i izazovi

Najveći izazovi bili su nedosledni nazivi kategorija, ponovljeni proizvodi i šifre/modeli ugrađeni u nazive. Zato se proveravaju i karakter-n-grami (dobri za nazive poput `kgv39vl31g`) umesto da se oslanjamo samo na cele reči. Linearni modeli su odabrani zbog brzine treniranja, ponovljivosti i dobre tačnosti na kratkim nazivima.
