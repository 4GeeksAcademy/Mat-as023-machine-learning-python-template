"""
WeLoveReviews — Análisis de Sentimiento en Reseñas de Clientes
Script de producción: carga el modelo, procesa reseñas y genera CSV con predicciones.

Uso:
    python src/app.py

Entrada: data/raw/reviews.csv
Salida:  data/processed/reviews_with_sentiment.csv
"""

import pandas as pd
import time
from transformers import pipeline
import torch

# ─── Configuración ─────────────────────────────────────────
MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
INPUT_PATH = "data/raw/reviews.csv"
OUTPUT_PATH = "data/processed/reviews_with_sentiment.csv"
BATCH_SIZE = 32

# ─── Funciones ─────────────────────────────────────────────

def load_model(model_id: str):
    """Carga el pipeline de sentiment-analysis UNA SOLA VEZ."""
    print(f"Cargando modelo: {model_id}")
    pipe = pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        max_length=512,
        truncation=True,
        device=0 if torch.cuda.is_available() else -1,
    )
    print("Modelo cargado exitosamente.")
    return pipe


def extract_stars(label: str) -> int:
    """Convierte '5 stars' → 5."""
    return int(label.split()[0])


def stars_to_sentiment(stars: int) -> str:
    """Mapea estrellas a banda de sentimiento."""
    if stars <= 2:
        return "Negativo"
    elif stars == 3:
        return "Neutral"
    else:
        return "Positivo"


def predict_batch(pipe, texts: list, batch_size: int = 32):
    """Ejecuta inferencia por lotes."""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_results = pipe(batch)
        results.extend(batch_results)
    return results


def process_dataframe(df: pd.DataFrame, pipe) -> pd.DataFrame:
    """
    Procesa el DataFrame completo: ejecuta inferencia y agrega columnas
    de predicción (estrellas, sentimiento, confianza).
    """
    texts = df["review_text"].tolist()
    print(f"Procesando {len(texts)} reseñas...")
    start = time.time()

    raw_results = predict_batch(pipe, texts, BATCH_SIZE)

    elapsed = time.time() - start
    print(f"Inferencia completada en {elapsed:.1f} segundos.")

    df_result = df.copy()
    df_result["predicted_label"] = [r["label"] for r in raw_results]
    df_result["predicted_stars"] = df_result["predicted_label"].apply(extract_stars)
    df_result["confidence"] = [r["score"] for r in raw_results]
    df_result["sentiment"] = df_result["predicted_stars"].apply(stars_to_sentiment)

    return df_result


def save_results(df: pd.DataFrame, path: str):
    """Guarda el DataFrame resultante a CSV sin índice accidental."""
    df.to_csv(path, index=False)
    print(f"Resultados guardados en: {path}")


# ─── Main ──────────────────────────────────────────────────

def main():
    # 1. Cargar modelo (UNA vez)
    pipe = load_model(MODEL_ID)

    # 2. Cargar datos
    print(f"Cargando datos desde: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")

    # 3. Limpieza básica
    df["review_text"] = df["review_text"].str.strip()

    # 4. Inferencia
    df_results = process_dataframe(df, pipe)

    # 5. Guardar
    save_results(df_results, OUTPUT_PATH)

    # 6. Resumen rápido
    print("\n=== RESUMEN ===")
    print(df_results["sentiment"].value_counts())
    print(f"\nRating humano promedio: {df_results['rating'].mean():.2f}★")
    print(f"Rating predicho promedio: {df_results['predicted_stars'].mean():.2f}★")


if __name__ == "__main__":
    main()
