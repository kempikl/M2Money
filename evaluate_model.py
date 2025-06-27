# evaluate_model.py

import os
import glob
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def detect_column(df: pd.DataFrame, patterns: list[str]) -> str:
    cols = df.columns
    lower = [c.lower() for c in cols]
    for pat in patterns:
        for idx, name in enumerate(lower):
            if pat in name:
                return cols[idx]
    raise KeyError(f"Nie znaleziono kolumny pasującej do wzorców: {patterns}")

def load_and_prepare(data_dir='data'):
    """Wczytuje wszystkie CSV (bez 'rent') i zwraca X, y."""
    all_files = sorted(p for p in glob.glob(os.path.join(data_dir, '*.csv'))
                       if 'rent' not in os.path.basename(p).lower())
    if not all_files:
        raise FileNotFoundError(f"Brak plików sprzedaży w katalogu {data_dir}")

    # detekcja kolumn na podstawie pierwszego pliku
    sample = pd.read_csv(all_files[0], nrows=5)
    city_col  = detect_column(sample, ["city"])
    area_col  = detect_column(sample, ["square", "sqm", "squaremeters"])
    rooms_col = detect_column(sample, ["rooms"])
    price_col = detect_column(sample, ["price"])

    dfs = []
    for path in all_files:
        df = pd.read_csv(path, usecols=[city_col, area_col, rooms_col, price_col])
        df.columns = ['city', 'area', 'rooms', 'price']
        dfs.append(df.dropna())
    data = pd.concat(dfs, ignore_index=True)
    return data[['city','area','rooms']], data['price']

def main():
    # 1. Przygotowanie danych i podział
    X, y = load_and_prepare('data')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Wczytanie modelu
    model_path = os.path.join('model','model.pkl')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Brak pliku modelu pod ścieżką {model_path}")
    pipeline = joblib.load(model_path)

    # 3. Predykcja na zbiorze testowym
    y_pred = pipeline.predict(X_test)

    # 4. Obliczenie metryk
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 5. Wyświetlenie wyników
    print("=== Ewaluacja modelu na zbiorze testowym ===")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:,.0f} PLN")
    print(f"RMSE : {rmse:,.0f} PLN")

    # 6. (Opcjonalnie) Zapis wyników do pliku CSV
    results = pd.DataFrame({
        'actual_price': y_test,
        'predicted_price': y_pred
    })
    os.makedirs('evaluation', exist_ok=True)
    out_path = os.path.join('evaluation','test_results.csv')
    results.to_csv(out_path, index=False)
    print(f"\nZapisano szczegóły predykcji do: {out_path}")

if __name__ == '__main__':
    main()

