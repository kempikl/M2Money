# --- Bazowy obraz z Pythonem ---
FROM python:3.10-slim

# --- Katalog roboczy ---
WORKDIR /app

# --- Kopiujemy zależności i instalujemy je ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Kopiujemy kod aplikacji i model ---
COPY . .

# --- Eksponujemy port Streamlit ---
EXPOSE 8501

# --- Domyślna komenda uruchamiająca Streamlit ---
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
