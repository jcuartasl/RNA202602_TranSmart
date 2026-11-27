"""CLI M1.

python -m transport_ai.forecasting.cli --mode full            # datos -> EDA -> búsqueda -> 30 semillas -> reporte (horas)
python -m transport_ai.forecasting.cli --mode demo            # < 2 min: artefactos congelados + mini-entrenamiento real
python -m transport_ai.forecasting.cli --mode forecast --route 50001-11001 --origin 2023-12-31 --quantile 0.9
Etapas sueltas: data | eda | search | seeds | report.  Cambios sin tocar código: --config otro.yaml o --set train.lr=0.003
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import yaml

from transport_ai.common.paths import FORECASTING_ARTIFACTS, ROOT
from transport_ai.forecasting.data import cargar_config


def aplicar_set(cfg: dict, pares: list[str]) -> dict:
    for par in pares:
        clave, valor = par.split("=", 1)
        *ruta, hoja = clave.split(".")
        nodo = cfg
        for k in ruta:
            nodo = nodo[k]
        if hoja not in nodo:
            raise KeyError(f"--set: clave desconocida {clave}")
        nodo[hoja] = yaml.safe_load(valor)
    return cfg


def etapa_search(cfg):
    from transport_ai.forecasting import train as T
    ctx = T.contexto(cfg)
    preds, tabla = T.correr_baselines(ctx, cfg)
    for s in ("val", "test"):
        pd.concat([preds[m][s] for m in preds]).to_parquet(T.EXP / f"pred_{s}_baselines.parquet", index=False)
    tablas = [tabla] + [T.buscar(a, ctx, cfg) for a in ("mlp", "gru")]
    pd.concat(tablas).to_csv(T.EXP / "search.csv", index=False)


def etapa_seeds(cfg, archs=("mlp", "gru")):
    from transport_ai.forecasting import train as T
    ctx = T.contexto(cfg)
    search = pd.read_csv(T.EXP / "search.csv")
    seeds = list(range(cfg["seeds"]["start"], cfg["seeds"]["start"] + cfg["seeds"]["n"]))
    for arch in archs:
        hp = json.loads(search[search["model"] == arch].sort_values("val_mase").iloc[0]["hp"])
        df, deploy = T.semillas(arch, hp, ctx, cfg, seeds)
        df.to_csv(T.EXP / f"seeds_{arch}.csv", index=False)
        for s, d in deploy.items():
            d.to_parquet(T.EXP / f"pred_{s}_{arch}.parquet", index=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["full", "demo", "data", "eda", "search", "seeds", "report", "forecast"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="CLAVE=VALOR")
    ap.add_argument("--arch", nargs="*", default=["mlp", "gru"])
    ap.add_argument("--route")
    ap.add_argument("--origin")
    ap.add_argument("--quantile", type=float, default=0.90)
    ap.add_argument("--model", default=None)
    a = ap.parse_args(argv)
    cfg = aplicar_set(cargar_config(a.config), a.set)
    t = time.time()
    FORECASTING_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ROOT / "experiments" / "m1").mkdir(parents=True, exist_ok=True)

    if a.mode in ("data", "full"):
        from transport_ai.forecasting.data import preparar
        print(json.dumps({k: v for k, v in preparar(cfg).items() if k != "archivos"}, indent=1, ensure_ascii=False))
    if a.mode in ("eda", "full"):
        from transport_ai.forecasting import eda
        print(eda.correr(cfg).to_string())
    if a.mode in ("search", "full"):
        etapa_search(cfg)
    if a.mode in ("seeds", "full"):
        etapa_seeds(cfg, a.arch)
    if a.mode in ("report", "full"):
        from transport_ai.forecasting import reporte
        print(reporte.correr(cfg).to_string())
    if a.mode == "demo":
        from transport_ai.forecasting import reporte
        reporte.demo(cfg, a.quantile)
    if a.mode == "forecast":
        from transport_ai.forecasting.infer import forecast
        print(forecast(a.route, a.origin, quantile=a.quantile, model=a.model).to_string(index=False))
    print(f"[m1] modo {a.mode} terminado en {time.time() - t:.1f} s")


if __name__ == "__main__":
    main()
