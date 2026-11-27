"""Construcción de series temporales diarias y features de calendario."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
import yaml
from datetime import date, timedelta

from transport_ai.common.paths import FORECASTING_PROCESSED
from transport_ai.common.logging_utils import get_logger

logger = get_logger(__name__)

# Festivos fijos Colombia (ley 51 de 1983 + adiciones)
# Formato: (mes, día): nombre
FESTIVOS_FIJOS = {
    (1, 1): "Año Nuevo",
    (1, 6): "Reyes Magos",
    (3, 19): "San José",
    (5, 1): "Día del Trabajo",
    (6, 29): "San Pedro y San Pablo",
    (7, 20): "Independencia Colombia",
    (8, 7): "Batalla de Boyacá",
    (8, 15): "Asunción de la Virgen",
    (10, 12): "Día de la Raza",
    (11, 1): "Todos los Santos",
    (11, 11): "Independencia de Cartagena",
    (12, 8): "Inmaculada Concepción",
    (12, 25): "Navidad",
}

# Festivos móviles (requieren cálculo por año)
# Se calculan dinámicamente basado en Semana Santa


def calcular_semana_santa(year: int) -> tuple[date, date]:
    """
    Calcula fechas de Jueves Santo y Viernes Santo para un año dado.
    Algoritmo de Gauss para fecha de Pascua.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1

    domingo_pascua = date(year, mes, dia)
    jueves_santo = domingo_pascua - timedelta(days=3)
    viernes_santo = domingo_pascua - timedelta(days=2)

    return jueves_santo, viernes_santo


def generar_calendario_anual(year: int) -> pd.DataFrame:
    """
    Genera calendario completo para un año con festivos, puentes y temporadas.

    Returns:
        DataFrame con columnas: fecha, es_festivo, festivo_nombre, es_puente,
        dia_semana, mes, dia_ano, es_fin_semana, temporada_escolar, semana_santa, fin_ano
    """
    # Rango completo del año
    fechas = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    df = pd.DataFrame({"fecha": fechas})

    # Básicos
    df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=Lun, 6=Dom
    df["mes"] = df["fecha"].dt.month
    df["dia_ano"] = df["fecha"].dt.dayofyear
    df["es_fin_semana"] = df["dia_semana"] >= 5

    # Festivos fijos
    df["es_festivo"] = False
    df["festivo_nombre"] = ""

    for (mes, dia), nombre in FESTIVOS_FIJOS.items():
        try:
            festivo_fecha = pd.Timestamp(year, mes, dia)
            mask = df["fecha"] == festivo_fecha
            df.loc[mask, "es_festivo"] = True
            df.loc[mask, "festivo_nombre"] = nombre
        except ValueError:
            pass  # Fecha inválida para ese año (ej. 29 feb)

    # Semana Santa (móvil)
    jueves, viernes = calcular_semana_santa(year)
    for d, nombre in [(jueves, "Jueves Santo"), (viernes, "Viernes Santo")]:
        mask = df["fecha"] == pd.Timestamp(d)
        df.loc[mask, "es_festivo"] = True
        df.loc[mask, "festivo_nombre"] = nombre

    # Puentes festivos: lunes siguiente a festivo en viernes o lunes
    # (Ley 51: si festivo cae sábado/domingo, se mueve a lunes siguiente)
    festivos_fecha = df.loc[df["es_festivo"], "fecha"]
    for f in festivos_fecha:
        if f.dayofweek == 4:  # Viernes -> puente lunes
            puente = f + timedelta(days=3)
            if puente.year == year:
                mask = df["fecha"] == puente
                df.loc[mask, "es_puente"] = True
                df.loc[mask, "festivo_nombre"] = "Puente: " + df.loc[df["fecha"] == f, "festivo_nombre"].values[0]
        elif f.dayofweek == 5:  # Sábado -> se mueve a lunes
            puente = f + timedelta(days=2)
            if puente.year == year:
                mask = df["fecha"] == puente
                df.loc[mask, "es_festivo"] = True
                df.loc[mask, "es_puente"] = True
                df.loc[mask, "festivo_nombre"] = "Traslado: " + df.loc[df["fecha"] == f, "festivo_nombre"].values[0]
        elif f.dayofweek == 6:  # Domingo -> se mueve a lunes
            puente = f + timedelta(days=1)
            if puente.year == year:
                mask = df["fecha"] == puente
                df.loc[mask, "es_festivo"] = True
                df.loc[mask, "es_puente"] = True
                df.loc[mask, "festivo_nombre"] = "Traslado: " + df.loc[df["fecha"] == f, "festivo_nombre"].values[0]

    df["es_puente"] = df["es_puente"].fillna(False).astype(bool)

    # Temporadas
    # Escolar: aprox. Feb-Jun y Ago-Nov (simplificado)
    df["temporada_escolar"] = df["mes"].isin([2, 3, 4, 5, 6, 8, 9, 10, 11])

    # Semana Santa: semana completa (domingo a sábado)
    if "jueves" in locals():
        inicio_ss = jueves - timedelta(days=3)  # Domingo de Ramos
        fin_ss = viernes + timedelta(days=2)    # Sábado de Gloria
        df["semana_santa"] = (df["fecha"] >= pd.Timestamp(inicio_ss)) & (df["fecha"] <= pd.Timestamp(fin_ss))
    else:
        df["semana_santa"] = False

    # Fin de año: 15 dic - 15 ene
    df["fin_ano"] = ((df["mes"] == 12) & (df["dia_ano"] >= 350)) | ((df["mes"] == 1) & (df["dia_ano"] <= 15))

    return df


def construir_calendario_completo(fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    """
    Construye calendario para rango de años.
    Args:
        fecha_inicio: "YYYY-MM-DD"
        fecha_fin: "YYYY-MM-DD"
    """
    inicio = pd.Timestamp(fecha_inicio)
    fin = pd.Timestamp(fecha_fin)

    años = range(inicio.year, fin.year + 1)
    calendarios = [generar_calendario_anual(a) for a in años]
    cal = pd.concat(calendarios, ignore_index=True)

    # Filtrar rango exacto
    cal = cal[(cal["fecha"] >= inicio) & (cal["fecha"] <= fin)].reset_index(drop=True)

    return cal


def construir_series_diarias(
    df_raw: pd.DataFrame,
    routes: list[str] = None,
    missing_policy: str = "zero",
    fecha_inicio: str = None,
    fecha_fin: str = None,
) -> pd.DataFrame:
    """
    Construye series diarias de pasajeros por ruta y sentido.

    Args:
        df_raw: DataFrame con columnas canónicas (fecha, ruta_id, pasajeros, etc.)
        routes: Lista de ruta_id a incluir. Si None, todas.
        missing_policy: 'zero' | 'interpolate' | 'flag' para días sin registros
        fecha_inicio, fecha_fin: Rango de fechas para la serie completa

    Returns:
        DataFrame con índice [fecha, ruta_id] y columna 'pasajeros'
    """
    if routes is not None:
        df_raw = df_raw[df_raw["ruta_id"].isin(routes)].copy()

    if fecha_inicio is None:
        fecha_inicio = df_raw["fecha"].min().strftime("%Y-%m-%d")
    if fecha_fin is None:
        fecha_fin = df_raw["fecha"].max().strftime("%Y-%m-%d")

    # Agregar pasajeros por ruta y día (suma si hay múltiples despachos)
    series = df_raw.groupby(["fecha", "ruta_id"])["pasajeros"].sum().reset_index()

    # Crear índice completo de todas las combinaciones fecha x ruta
    fechas_completas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    rutas_unicas = series["ruta_id"].unique()

    idx_completo = pd.MultiIndex.from_product(
        [fechas_completas, rutas_unicas],
        names=["fecha", "ruta_id"]
    )

    # Reindexar
    series_indexed = series.set_index(["fecha", "ruta_id"]).reindex(idx_completo)

    # Política de días sin registros
    if missing_policy == "zero":
        # Asumir 0 pasajeros (día sin operación)
        series_indexed["pasajeros"] = series_indexed["pasajeros"].fillna(0).astype(int)
    elif missing_policy == "interpolate":
        # Interpolar lineal por ruta
        series_indexed["pasajeros"] = (
            series_indexed.groupby(level="ruta_id")["pasajeros"]
            .apply(lambda x: x.interpolate(method="time").ffill().bfill())
            .astype(int)
        )
    elif missing_policy == "flag":
        # Marcar como NaN para tratamiento posterior
        series_indexed["pasajeros"] = series_indexed["pasajeros"].astype("Int64")  # nullable int
    else:
        raise ValueError(f"missing_policy debe ser 'zero', 'interpolate' o 'flag', no '{missing_policy}'")

    series_final = series_indexed.reset_index()
    series_final = series_final.rename(columns={"pasajeros": "pasajeros_dia"})

    logger.info(f"Series construidas: {len(series_final)} registros, {len(rutas_unicas)} rutas")
    logger.info(f"Rango: {fecha_inicio} a {fecha_fin}")

    return series_final


def detectar_anomalias(
    series: pd.DataFrame,
    max_pasajeros_por_despacho: int = 100,
    umbral_cero: float = 0.05,
) -> pd.DataFrame:
    """
    Detecta anomalías en las series.

    Args:
        series: DataFrame con columnas fecha, ruta_id, pasajeros_dia
        max_pasajeros_por_despacho: Umbral para valores extremos
        umbral_cero: Fracción máxima de días con 0 permitida por ruta

    Returns:
        DataFrame con columna 'anomalia' (True/False) y 'tipo_anomalia'
    """
    df = series.copy()
    df["anomalia"] = False
    df["tipo_anomalia"] = ""

    # Por ruta
    for ruta in df["ruta_id"].unique():
        mask_ruta = df["ruta_id"] == ruta
        vals = df.loc[mask_ruta, "pasajeros_dia"]

        # 1. Valores extremos (posibles errores de digitación)
        extremos = vals > max_pasajeros_por_despacho * 10  # Heurística
        df.loc[mask_ruta & extremos, "anomalia"] = True
        df.loc[mask_ruta & extremos, "tipo_anomalia"] = "valor_extremo"

        # 2. Exceso de ceros (ruta inactiva o error)
        frac_ceros = (vals == 0).mean()
        if frac_ceros > umbral_cero:
            # Marcar solo si no es una ruta legítimamente estacional
            ceros_mask = (vals == 0) & mask_ruta
            df.loc[ceros_mask, "anomalia"] = True
            df.loc[ceros_mask, "tipo_anomalia"] = "exceso_ceros"

    n_anom = df["anomalia"].sum()
    logger.info(f"Anomalías detectadas: {n_anom} ({n_anom/len(df)*100:.1f}%)")

    return df


def agregar_features_calendario(
    series: pd.DataFrame,
    calendario: pd.DataFrame = None,
    fecha_inicio: str = None,
    fecha_fin: str = None,
) -> pd.DataFrame:
    """
    Agrega variables de calendario a las series.

    Args:
        series: DataFrame con columnas fecha, ruta_id, pasajeros_dia
        calendario: DataFrame opcional pre-generado
        fecha_inicio, fecha_fin: Para generar calendario si no se provee

    Returns:
        DataFrame con features de calendario mergeadas
    """
    if calendario is None:
        if fecha_inicio is None:
            fecha_inicio = series["fecha"].min().strftime("%Y-%m-%d")
        if fecha_fin is None:
            fecha_fin = series["fecha"].max().strftime("%Y-%m-%d")
        calendario = construir_calendario_completo(fecha_inicio, fecha_fin)

    # Merge
    df = series.merge(calendario, on="fecha", how="left")

    # Features cíclicas (seno/coseno para día de semana, día de año)
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["dia_ano_sin"] = np.sin(2 * np.pi * df["dia_ano"] / 365.25)
    df["dia_ano_cos"] = np.cos(2 * np.pi * df["dia_ano"] / 365.25)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    # Features binarias útiles
    df["es_festivo_o_puente"] = df["es_festivo"] | df["es_puente"]
    df["es_temporada_alta"] = df["semana_santa"] | df["fin_ano"]

    logger.info(f"Features de calendario agregadas: {len(df.columns)} columnas totales")

    return df


def crear_ventanas_deslizantes(
    series: pd.DataFrame,
    window_size: int = 30,
    horizon: int = 30,
    target_col: str = "pasajeros_dia",
    feature_cols: list[str] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Crea ventanas deslizantes para entrenamiento (X, y) sin fuga temporal.

    Args:
        series: DataFrame con MultiIndex [fecha, ruta_id] o columnas fecha, ruta_id
        window_size: Días de historia como entrada
        horizon: Días a predecir (multi-output)
        target_col: Columna objetivo
        feature_cols: Columnas de features adicionales (None = solo target)

    Returns:
        X: (n_samples, window_size, n_features)
        y: (n_samples, horizon)
        rutas: array de ruta_id por muestra
        metadata: DataFrame con info de cada ventana (fecha_origen, ruta, etc.)
    """
    df = series.copy()

    # Asegurar índice ordenado
    if "fecha" in df.columns and "ruta_id" in df.columns:
        df = df.sort_values(["ruta_id", "fecha"]).reset_index(drop=True)

    if feature_cols is None:
        # Usar target + features de calendario binarias/cíclicas
        feature_cols = [target_col] + [
            c for c in df.columns
            if c.startswith(("dia_semana_", "dia_ano_", "mes_", "es_", "semana_", "fin_", "temporada_"))
        ]

    # Filtrar columnas existentes
    feature_cols = [c for c in feature_cols if c in df.columns]

    X_list, y_list, rutas_list, meta_list = [], [], [], []

    for ruta in df["ruta_id"].unique():
        df_ruta = df[df["ruta_id"] == ruta].reset_index(drop=True)
        n = len(df_ruta)

        if n < window_size + horizon:
            logger.warning(f"Ruta {ruta}: pocos datos ({n} < {window_size + horizon}), saltando")
            continue

        # Ventanas deslizantes: cada día es un origen posible
        for i in range(window_size, n - horizon + 1):
            # Ventana de entrada: [i-window_size, i)
            X_window = df_ruta.iloc[i - window_size:i][feature_cols].values
            # Ventana de salida: [i, i+horizon)
            y_window = df_ruta.iloc[i:i + horizon][target_col].values

            X_list.append(X_window)
            y_list.append(y_window)
            rutas_list.append(ruta)
            meta_list.append({
                "fecha_origen": df_ruta.iloc[i - 1]["fecha"],
                "ruta_id": ruta,
                "window_start": df_ruta.iloc[i - window_size]["fecha"],
                "window_end": df_ruta.iloc[i - 1]["fecha"],
                "target_start": df_ruta.iloc[i]["fecha"],
                "target_end": df_ruta.iloc[i + horizon - 1]["fecha"],
            })

    if not X_list:
        raise ValueError("No se pudieron crear ventanas. Verificar window_size, horizon y datos.")

    X = np.stack(X_list)  # (n_samples, window_size, n_features)
    y = np.stack(y_list)  # (n_samples, horizon)
    rutas = np.array(rutas_list)
    metadata = pd.DataFrame(meta_list)

    logger.info(f"Ventanas creadas: {X.shape[0]} muestras, X={X.shape}, y={y.shape}")

    return X, y, rutas, metadata


def cargar_config() -> dict:
    """Carga configuración forecast.yaml."""
    from transport_ai.common.paths import CONFIGS
    with open(CONFIGS / "forecast.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)