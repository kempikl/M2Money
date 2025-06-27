import streamlit as st
import pandas as pd
import joblib
import unicodedata

# Funkcja do normalizacji nazwy miasta (lowercase + usunięcie polskich znaków)
def normalize_city(name: str) -> str:
    name_lower = name.lower()
    normalized = unicodedata.normalize('NFKD', name_lower)
    return ''.join([c for c in normalized if not unicodedata.combining(c)])

# Wczytanie wytrenowanego modelu i pipeline
model_pipeline = joblib.load("model/model.pkl")

# Pobranie listy miast z nauczonego OneHotEncoder w pipeline'u
encoder = model_pipeline.named_steps['prep'].named_transformers_['cat']
model_cities = encoder.categories_[0].tolist()  # nazwy miast w formie z modelu
display_cities = [c.capitalize() for c in model_cities]  # na potrzeby UI

st.title("M2Money – Prognoza cen mieszkań")

# Pola wejściowe dla użytkownika
selected_city = st.selectbox("Miasto:", display_cities)
area = st.number_input("Powierzchnia [m²]:", min_value=10.0, max_value=500.0, value=50.0, step=1.0)
rooms = st.number_input("Liczba pokoi:", min_value=1, max_value=10, value=2, step=1)
condition = st.selectbox("Stan mieszkania:", ["po remoncie", "do remontu"])

# Normalizacja wejścia
model_city = normalize_city(selected_city)

# Walidacja danych
errors = False
if area < 10 or area > 500:
    st.error("Powierzchnia musi być między 10 a 500 m².")
    errors = True
if rooms < 1 or rooms > 10:
    st.error("Liczba pokoi musi być między 1 a 10.")
    errors = True

# Funkcja korekty stanu

def get_corrected_price(base: float, condition_state: str) -> float:
    return base * 0.8 if condition_state == "do remontu" else base

# Obliczenia po kliknięciu
if st.button("Oblicz wycenę"):
    if not errors:
        # Przygotowanie danych dla modelu
        input_df = pd.DataFrame([{ 'city': model_city, 'area': area, 'rooms': rooms }])
        base_price = model_pipeline.predict(input_df)[0]

        # Wyliczenie ceny dla obu stanów
        selected_price = get_corrected_price(base_price, condition)
        opposite_state = "po remoncie" if condition == "do remontu" else "do remontu"
        opposite_price = get_corrected_price(base_price, opposite_state)

        selected_price = int(selected_price)
        opposite_price = int(opposite_price)
        state_diff = opposite_price - selected_price

        # Wyświetlenie wyników
        st.subheader(f"Wycena ({condition}): {selected_price:,d} PLN")
        st.subheader(f"Wycena ({opposite_state}): {opposite_price:,d} PLN")
        st.write(f"**Różnica między stanami:** {state_diff:,d} PLN")

        # Porównanie do innych miast
        comparison = []
        for disp in display_cities:
            if disp == selected_city:
                continue
            rep_city_norm = normalize_city(disp)
            rep_df = pd.DataFrame([{ 'city': rep_city_norm, 'area': area, 'rooms': rooms }])
            rep_base = model_pipeline.predict(rep_df)[0]
            rep_price = get_corrected_price(rep_base, condition)
            rep_price = int(rep_price)
            diff = rep_price - selected_price
            comparison.append({ 'Miasto': disp, 'Wycena [PLN]': rep_price, 'Różnica [PLN]': diff })

        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df.sort_values(by='Różnica [PLN]', ascending=False).reset_index(drop=True)
        st.subheader("Porównanie do innych miast")
        st.dataframe(comp_df)
    else:
        st.warning("Popraw błędy w danych wejściowych przed ponowną próbą.")
