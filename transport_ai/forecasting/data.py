"""Datos M1: agregación del CSV oficial, auditoría de calidad y series diarias por ruta dirigida.

Fuente: Superintendencia de Transporte, "Operación de pasajeros y despacho de vehículos en la
modalidad de transporte de pasajeros por carretera" (datos.gov.co, eh75-8ah6, CC BY-SA 4.0).
Ver data/processed/forecasting/DICCIONARIO.md.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from transport_ai.common.paths import CONFIGS, FORECASTING_PROCESSED, ROOT

COLS = ["TERMINAL", "CLASE_VEHICULO", "NIVEL_SERVICIO", "MUNICIPIO_ORIGEN_RUTA",
        "MUNICIPIO_DESTINO_RUTA", "FECHA_DESPACHO", "HORA_DESPACHO", "TIPO_DESPACHO",
        "DESPACHOS", "PASAJEROS"]
KEY = ["fecha", "origen", "destino", "tipo"]


def cargar_config(path: str | Path | None = None) -> dict:
    with open(path or CONFIGS / "forecast.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def agregar_crudo(raw_path: Path, out_dir: Path = FORECASTING_PROCESSED,
                  cap_por_despacho: int = 80, chunksize: int = 2_000_000) -> dict:
    """Una pasada por bloques sobre el CSV crudo (~33 M filas).

    Escribe agregados diarios por ruta/tipo, reporte de terminales por día y una tabla de anomalías.
    No descarta nada: solo marca. Las decisiones de limpieza se toman al construir la serie.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    partes, terminales, anomalias, hashes = [], [], [], []
    n_filas = 0
    for ch in pd.read_csv(raw_path, usecols=COLS, dtype=str, chunksize=chunksize):
        n_filas += len(ch)
        # hash de fila completa para detectar duplicados exactos entre bloques
        hashes.append(pd.util.hash_pandas_object(ch, index=False).to_numpy())
        fecha = pd.to_datetime(ch["FECHA_DESPACHO"].str[:10], format="%m/%d/%Y", errors="coerce")
        pax = pd.to_numeric(ch["PASAJEROS"], errors="coerce")
        desp = pd.to_numeric(ch["DESPACHOS"], errors="coerce")
        df = pd.DataFrame({
            "fecha": fecha,
            "origen": ch["MUNICIPIO_ORIGEN_RUTA"].str.zfill(5),  # DIVIPOLA como texto
            "destino": ch["MUNICIPIO_DESTINO_RUTA"].str.zfill(5),
            "tipo": ch["TIPO_DESPACHO"],
            "terminal": ch["TERMINAL"],
            "pasajeros": pax, "despachos": desp,
        })
        malas = df["fecha"].isna() | (df["fecha"].dt.year > 2026) | pax.isna() | desp.isna() | (pax < 0) | (desp < 0)
        if malas.any():
            a = ch[malas].copy()
            a["motivo"] = "fecha_imposible_o_valor_invalido"
            anomalias.append(a)
        df = df[~malas]
        df["reg_pax0"] = ((df["pasajeros"] == 0) & (df["despachos"] > 0)).astype(int)
        exceso = df["pasajeros"] > cap_por_despacho * df["despachos"].clip(lower=1)
        df["reg_exceso"] = exceso.astype(int)
        df["pax_exceso"] = df["pasajeros"].where(exceso, 0)
        df["registros"] = 1
        partes.append(df.groupby(KEY, as_index=False)[
            ["pasajeros", "despachos", "registros", "reg_pax0", "reg_exceso", "pax_exceso"]].sum())
        terminales.append(df.groupby(["fecha", "terminal"], as_index=False)["registros"].sum())

    diario = pd.concat(partes).groupby(KEY, as_index=False).sum()
    term = pd.concat(terminales).groupby(["fecha", "terminal"], as_index=False).sum()
    h = np.concatenate(hashes)
    n_dup = int(len(h) - len(np.unique(h)))
    diario.to_parquet(out_dir / "diario_agregado.parquet", index=False)
    term.to_parquet(out_dir / "terminales_dia.parquet", index=False)
    anom = pd.concat(anomalias) if anomalias else pd.DataFrame(columns=COLS + ["motivo"])
    anom.to_csv(out_dir / "anomalias_crudo.csv", index=False)
    resumen = {
        "raw_path": str(Path(raw_path).relative_to(ROOT)) if Path(raw_path).is_absolute() else str(raw_path),
        "raw_sha256": sha256(Path(raw_path)),
        "filas": n_filas,
        "filas_invalidas": int(len(anom)),
        "duplicados_exactos": n_dup,
        "registros_pax0_con_despacho": int(diario["reg_pax0"].sum()),
        "registros_exceso_capacidad": int(diario["reg_exceso"].sum()),
        "cap_pasajeros_por_despacho": cap_por_despacho,
        "fecha_min": str(diario["fecha"].min().date()),
        "fecha_max": str(diario["fecha"].max().date()),
        "registros_por_anio": {str(k): int(v) for k, v in
                               diario.groupby(diario["fecha"].dt.year)["registros"].sum().items()},
        "rutas_dirigidas": int(diario[["origen", "destino"]].drop_duplicates().shape[0]),
    }
    (out_dir / "auditoria_crudo.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    return resumen


def serie_completa(diario: pd.DataFrame, inicio: str, fin: str, tipo: str = "ORIGEN",
                   excluir_exceso: bool = True) -> pd.DataFrame:
    """Serie diaria continua por ruta dirigida (origen→destino) en [inicio, fin].

    `pasajeros` es NaN cuando el día es desconocido (sin registros, o cero pasajeros con despachos > 0);
    no se rellena con cero. `pasajeros_sin_limpiar` guarda la versión sin tratamiento (sensibilidad).
    """
    d = diario[(diario["tipo"] == tipo) & diario["fecha"].between(inicio, fin)].copy()
    d["route_id"] = d["origen"] + "-" + d["destino"]
    d["pasajeros_sin_limpiar"] = d["pasajeros"]
    if excluir_exceso:  # registros con más pasajeros que capacidad física: se retiran del conteo
        d["pasajeros"] = d["pasajeros"] - d["pax_exceso"]
    fechas = pd.date_range(inicio, fin, freq="D")
    idx = pd.MultiIndex.from_product([d["route_id"].unique(), fechas], names=["route_id", "fecha"])
    cols = ["pasajeros", "pasajeros_sin_limpiar", "despachos", "registros", "reg_pax0", "reg_exceso"]
    s = d.set_index(["route_id", "fecha"])[cols].reindex(idx).reset_index()
    s["anomalia"] = ""
    s.loc[s["registros"].isna(), "anomalia"] = "sin_registro"
    # cero pasajeros con despachos > 0: pasajeros no registrados, no demanda cero -> desconocido
    cero = (s["pasajeros"] == 0) & (s["despachos"] > 0)
    s.loc[cero, "anomalia"] = "cero_con_despacho"
    s.loc[cero, "pasajeros"] = np.nan
    s["valido"] = s["pasajeros"].notna()
    return s


def cobertura(series: pd.DataFrame, terminales: pd.DataFrame, min_terminales_frac: float = 0.5) -> tuple[pd.DataFrame, pd.Series]:
    """Cobertura por ruta + días de apagón nacional de reporte.

    Un día es 'apagón' si reportan menos de `min_terminales_frac` de la mediana de terminales activas:
    en esos días la ausencia de registros es falta de reporte, no demanda cero.
    """
    n_term = terminales.groupby("fecha")["terminal"].nunique()
    n_term = n_term.reindex(pd.date_range(series["fecha"].min(), series["fecha"].max()), fill_value=0)
    apagon = n_term < min_terminales_frac * n_term.median()
    g = series.groupby("route_id")
    cob = pd.DataFrame({
        "dias": g.size(),
        "dias_validos": g["valido"].sum(),
        "pasajeros_total": g["pasajeros"].sum(),
        "pasajeros_media": g["pasajeros"].mean(),
        "ceros_validos": g["pasajeros"].apply(lambda x: int((x == 0).sum())),
        "dias_cero_con_despacho": g["anomalia"].apply(lambda x: int((x == "cero_con_despacho").sum())),
    })
    cob["cobertura"] = cob["dias_validos"] / cob["dias"]
    # mayor tramo consecutivo sin registros (huecos largos invalidan ventanas de entrada)
    cob["max_hueco"] = g["valido"].apply(lambda o: int((~o).groupby(o.cumsum()).sum().max()))
    return cob.sort_values("pasajeros_total", ascending=False).reset_index(), apagon


def seleccionar_rutas(cob: pd.DataFrame, min_cobertura: float, max_hueco: int,
                      min_media: float, n_rutas: int) -> pd.DataFrame:
    """Rutas aptas = cobertura alta, sin huecos largos y volumen suficiente; se toman las de mayor volumen."""
    cob = cob.copy()
    cob["apta"] = (cob["cobertura"] >= min_cobertura) & (cob["max_hueco"] <= max_hueco) & (cob["pasajeros_media"] >= min_media)
    cob["seleccionada"] = False
    cob.loc[cob[cob["apta"]].index[:n_rutas], "seleccionada"] = True
    return cob


def preparar(cfg: dict) -> dict:
    """Crudo -> series diarias, cobertura, selección de rutas, anomalías y manifiesto (reproducible)."""
    c, pdir = cfg["data"], ROOT / cfg["data"]["processed_dir"]
    if not (pdir / "diario_agregado.parquet").exists():
        agregar_crudo(ROOT / c["raw_path"], pdir, c["cap_pasajeros_por_despacho"])
    diario = pd.read_parquet(pdir / "diario_agregado.parquet")
    term = pd.read_parquet(pdir / "terminales_dia.parquet")
    s = serie_completa(diario, c["inicio"], c["fin"], c["tipo_despacho"])
    cob, apagon = cobertura(s, term)
    if apagon.any():  # días sin reporte nacional: desconocidos aunque haya algún registro suelto
        s.loc[s["fecha"].isin(apagon[apagon].index), ["pasajeros", "valido"]] = [np.nan, False]
    rutas = seleccionar_rutas(cob, **c["seleccion"])
    # cruce DIVIPOLA (DANE): códigos como texto para conservar ceros iniciales
    dv = pd.read_csv(ROOT / c["divipola_path"], dtype=str).set_index("cod_mpio")
    for k, lado in enumerate(("origen", "destino")):
        cod = rutas["route_id"].str.split("-").str[k]
        rutas[f"{lado}_nombre"], rutas[f"{lado}_dpto"] = cod.map(dv["nom_mpio"]), cod.map(dv["dpto"])
    sel = rutas.loc[rutas["seleccionada"], "route_id"]
    serie = s[s["route_id"].isin(sel)].sort_values(["route_id", "fecha"]).reset_index(drop=True)
    # atípicos: muy lejos de la mediana móvil de 28 días de la ruta; se marcan pero se conservan
    med = serie.groupby("route_id")["pasajeros"].transform(lambda x: x.rolling(28, center=True, min_periods=14).median())
    ratio = serie["pasajeros"] / med
    serie["atipico"] = (ratio < 0.2) | (ratio > 5)
    serie.to_parquet(pdir / "series.parquet", index=False)
    rutas.to_parquet(pdir / "rutas.parquet", index=False)
    serie[(serie["anomalia"] != "") | serie["atipico"]].to_csv(pdir / "anomalias_series.csv", index=False)
    auditoria = json.loads((pdir / "auditoria_crudo.json").read_text(encoding="utf-8"))
    manifiesto = {
        "fuente": "Superintendencia de Transporte - datos.gov.co: Operación de pasajeros y despacho de vehículos en la modalidad de transporte de pasajeros por carretera",
        "raw_sha256": auditoria["raw_sha256"], "raw_filas": auditoria["filas"],
        "periodo": [c["inicio"], c["fin"]], "tipo_despacho": c["tipo_despacho"],
        "rutas_totales": int(len(rutas)), "rutas_aptas": int(rutas["apta"].sum()), "rutas_seleccionadas": sel.tolist(),
        "dias_apagon_reporte": int(apagon.sum()),
        "rutas_con_codigo_fuera_de_divipola": int(rutas[["origen_nombre", "destino_nombre"]].isna().any(axis=1).sum()), "splits": cfg["splits"],
        "archivos": {f.name: sha256(f) for f in sorted(pdir.glob("*")) if f.suffix in (".parquet", ".csv", ".json")
                     and f.name not in ("diario_agregado.parquet", "terminales_dia.parquet")},
    }
    out = ROOT / "data" / "manifests" / "forecast_manifest.json"
    out.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifiesto
