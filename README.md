---
title: "M2Money"
emoji: "🏠"
colorFrom: "blue"
colorTo: "green"
sdk: streamlit
sdk_version: "1.24.0"
app_file: app.py
pinned: false
---

# PJATK SUML Project – M2Money

Link do aplikacji na Hugging Face Spaces: https://huggingface.co/spaces/kempikl/M2Money

## 🎯 Cel projektu

* Zapewnienie narzędzia dla użytkowników (kupujących, sprzedających, inwestorów) do szybkiej wyceny nieruchomości.
* Automatyzacja trenowania i wdrażania modelu ML przy użyciu Docker i CI/CD (GitHub Actions → Hugging Face Spaces).

---

## 🚀 Funkcjonalności

1. Interaktywny formularz Streamlit:

   * Wybór miasta, metrażu, liczby pokoi i stanu technicznego (po remoncie / do remontu).
   * Walidacja wartości metrażu i liczby pokoi.
   * Normalizacja nazw miast (lowercase + usunięcie polskich znaków).
2. Model ML:

   * Pipeline: OneHotEncoder(miasto) + StandardScaler(metraż, pokoje) + RandomForestRegressor.
   * Porównanie wartości mieszkania w stanie wyjściowym i po zmianie stanu.
3. Porównanie do innych miast:

   * Dynamiczny odczyt listy miast z wytrenowanego OneHotEncoder.
   * Tabela różnic cenowych.
4. Konteneryzacja w Dockerze.
5. CI/CD:

   * GitHub Actions automatycznie wdraża aplikację do Hugging Face Spaces przy każdym pushu do `main`.

---

## ⚙️ Instalacja i uruchomienie

### 1. Klonowanie repozytorium

```bash
git clone https://github.com/kempikl/M2Money.git
cd M2Money
```

### 2a. Uruchomienie lokalne z pip

```bash
python -m venv venv
source venv/bin/activate      # Linux/MacOS
# venv\Scripts\activate     # Windows
pip install --no-cache-dir -r requirements.txt

# (Opcjonalnie) trenowanie modelu na nowo:
python train_model.py

# Uruchomienie aplikacji:
streamlit run app.py
```

### 2b. Uruchomienie z Condą

```bash
conda env create -f environment.yml
conda activate m2money

python train_model.py
streamlit run app.py
```

### 3. Uruchomienie w Dockerze

```bash
docker build -t m2money:latest .
docker run -p 8501:8501 m2money:latest
```

---

## 🔍 Szczegóły techniczne

* **Dane:** [Kaggle: Apartment Prices in Poland](https://www.kaggle.com/datasets/krzysztofjamroz/apartment-prices-in-poland/data)
* **Model:** RandomForestRegressor + proste cechy (miasto, powierzchnia, pokoje)
* **Deployment:** Hugging Face Spaces (port 8501)

---

## 📈 Wyniki ewaluacji

* R²: \~0.83
* MAE: \~98 000 PLN
* RMSE: \~165 000 PLN
