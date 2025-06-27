# train_model.py

import os
import glob
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

def detect_column(df: pd.DataFrame, patterns: list[str]) -> str:
    """
    Znajdź kolumnę, której nazwa zawiera dowolny z wzorców (case-insensitive).
    """
    cols = df.columns
    lower = [c.lower() for c in cols]
    for pat in patterns:
        for idx, name in enumerate(lower):
            if pat in name:
                return cols[idx]
    raise KeyError(f"Nie znaleziono kolumny pasującej do wzorców: {patterns}")

def load_and_prepare(data_dir='data'):
    # weź tylko pliki sprzedaży (pomijamy te z 'rent' w nazwie)
    all_files = sorted(p for p in glob.glob(os.path.join(data_dir, '*.csv'))
                       if 'rent' not in os.path.basename(p).lower())
    if not all_files:
        raise FileNotFoundError(f"Brak plików sprzedaży w katalogu {data_dir}")

    # podstawowe detekcje na pierwszym pliku
    sample = pd.read_csv(all_files[0], nrows=5)
    print(">>> Kolumny w pierwszym pliku:", list(sample.columns))
    city_col  = detect_column(sample, ["city"])
    area_col  = detect_column(sample, ["square", "sqm", "squaremeters"])
    rooms_col = detect_column(sample, ["rooms"])
    price_col = detect_column(sample, ["price"])  # to jest cena całkowita

    print(f">>> Użyte kolumny: city='{city_col}', area='{area_col}', rooms='{rooms_col}', price='{price_col}'")

    dfs = []
    for path in all_files:
        df = pd.read_csv(path)
        # wybieramy i standaryzujemy nazwy
        df = df[[city_col, area_col, rooms_col, price_col]]
        df.columns = ['city', 'area', 'rooms', 'price']
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    data = data.dropna(subset=['city', 'area', 'rooms', 'price'])
    return data[['city', 'area', 'rooms']], data['price']

def main():
    # 1. Wczytanie i przygotowanie danych
    X, y = load_and_prepare('data')

    # 2. Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Pipeline: one-hot dla miasta + scaling dla area/rooms + model lasu losowego
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), ['city']),
        ('num', StandardScaler(), ['area', 'rooms'])
    ])
    pipeline = Pipeline([
        ('prep', preprocessor),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # 4. Trenowanie i ewaluacja
    pipeline.fit(X_train, y_train)
    print(f"R² trening: {pipeline.score(X_train, y_train):.3f}")
    print(f"R² test:    {pipeline.score(X_test, y_test):.3f}")

    # 5. Zapis pipeline
    os.makedirs('model', exist_ok=True)
    joblib.dump(pipeline, 'model/model.pkl')
    print("Model zapisany jako model/model.pkl")

if __name__ == '__main__':
    main()
