"""Entrenamiento M1: referencias, búsqueda aleatoria de hiperparámetros (solo validación) y 30 semillas.

Protocolo: mismas rutas, mismo escalado (media de entrenamiento por ruta), mismos orígenes diarios en
val/test y mismas métricas para los cuatro modelos. Test solo se lee después de congelar configuraciones.
Los pesos se ajustan una vez (train + parada temprana en val); en cada origen rodante solo se actualiza
la ventana de entrada (operación sin reentrenamiento entre orígenes).
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import torch

from transport_ai.common.paths import EXPERIMENTS, FORECASTING_ARTIFACTS, ROOT
from transport_ai.common.seeds import set_seed
from transport_ai.forecasting import baselines as B
from transport_ai.forecasting import evaluation as E
from transport_ai.forecasting.features import CAL_COLS, calendario, origenes, panel, ventanas
from transport_ai.forecasting.models import crear, perdida

EXP = EXPERIMENTS / "m1"


def contexto(cfg: dict) -> dict:
    """Panel de las rutas seleccionadas, escalado por la media de ENTRENAMIENTO de cada ruta."""
    pdir = ROOT / cfg["data"]["processed_dir"]
    rutas = pd.read_parquet(pdir / "rutas.parquet").query("seleccionada")["route_id"].tolist()
    Y, fechas = panel(pd.read_parquet(pdir / "series.parquet"), rutas)
    a, b = cfg["splits"]["train"]
    scale = np.nanmean(Y[:, (fechas >= a) & (fechas <= b)], axis=1)
    return {"Y": Y, "Ys": Y / scale[:, None], "fechas": fechas, "rutas": rutas, "scale": scale,
            "cal": calendario(fechas).to_numpy(), "levels": cfg["features"]["quantiles"],
            "H": cfg["features"]["horizon"], "denom": dict(zip(rutas, E.denominador_mase(Y, fechas, a, b)))}


def datos(ctx: dict, cfg: dict, window: int) -> dict:
    return {s: ventanas(ctx["Ys"], ctx["cal"], origenes(ctx["fechas"], *cfg["splits"][s], window, ctx["H"]),
                        window, ctx["H"]) for s in ("train", "val", "test")}


def largo(ctx, d, point, q, modelo):
    return E.tabla_larga(d, point, q, ctx["levels"], ctx["rutas"], ctx["fechas"], ctx["scale"], modelo)


# ---------------------------------------------------------------- referencias

def correr_baselines(ctx: dict, cfg: dict) -> tuple[dict, pd.DataFrame]:
    """Ingenuo estacional + Ridge (búsqueda alpha × ventana en validación). Devuelve tablas largas val/test."""
    H, lv, n_r = ctx["H"], ctx["levels"], len(ctx["rutas"])
    out, filas = {}, []
    D = datos(ctx, cfg, 28)
    pts = {s: B.naive_estacional(ctx["Ys"], D[s]["route"], D[s]["t0"], H) for s in ("val", "test")}
    off = B.offsets_cuantiles(D["val"]["y"], pts["val"], lv)
    out["naive"] = {s: largo(ctx, D[s], pts[s], B.con_cuantiles(pts[s], off), "naive") for s in ("val", "test")}
    art = {"naive": {"offsets": off}}

    grid = list(itertools.product(cfg["search"]["ridge"]["alpha"], cfg["search"]["ridge"]["window"]))
    random.Random(cfg["search"]["seed"]).shuffle(grid)
    mejor = None
    for alpha, w in grid[:cfg["search"]["budget"]["ridge"]]:
        t = time.time()
        D = datos(ctx, cfg, w)
        m = B.ajustar_ridge(D["train"], n_r, alpha)
        pv = np.clip(m.predict(B.matriz(D["val"], n_r)), 0, None)
        met = E.metricas(largo(ctx, D["val"], pv, B.con_cuantiles(pv, np.zeros((H, len(lv)))), "ridge"), ctx["denom"])
        filas.append({"model": "ridge", "hp": json.dumps({"alpha": alpha, "window": w}), "val_mase": met["mase"],
                      "val_mae": met["mae"], "params": int(m.coef_.size + m.intercept_.size), "seconds": time.time() - t})
        if mejor is None or met["mase"] < mejor[0]:
            mejor = (met["mase"], alpha, w, m, D, pv)
    _, alpha, w, m, D, pv = mejor
    off = B.offsets_cuantiles(D["val"]["y"], pv, lv)
    pt = np.clip(m.predict(B.matriz(D["test"], n_r)), 0, None)
    out["ridge"] = {"val": largo(ctx, D["val"], pv, B.con_cuantiles(pv, off), "ridge"),
                    "test": largo(ctx, D["test"], pt, B.con_cuantiles(pt, off), "ridge")}
    art["ridge"] = {"model": m, "offsets": off, "hp": {"alpha": alpha, "window": w}}
    joblib.dump(art, FORECASTING_ARTIFACTS / "baselines.joblib")
    return out, pd.DataFrame(filas)


# ---------------------------------------------------------------- redes

def tensores(d: dict) -> list[torch.Tensor]:
    return [torch.from_numpy(d["past"]), torch.from_numpy(d["fut"]), torch.from_numpy(d["route"]).long(),
            torch.from_numpy(d["y"])]


def predecir(model, d: dict, bs: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    past, fut, route, _ = tensores(d)
    with torch.no_grad():
        outs = [model(past[i:i + bs], fut[i:i + bs], route[i:i + bs]) for i in range(0, len(route), bs)]
    return torch.cat([o[0] for o in outs]).numpy(), torch.cat([o[1] for o in outs]).numpy()


def construir(arch: str, hp: dict, ctx: dict):
    kw = {k: v for k, v in hp.items() if k not in ("window", "lr", "loss")}
    return crear(arch, window=hp["window"], n_feat=2 + len(CAL_COLS), n_cal=len(CAL_COLS), horizon=ctx["H"],
                 n_routes=len(ctx["rutas"]), n_q=len(ctx["levels"]), **kw)


def ajustar(arch: str, hp: dict, D: dict, ctx: dict, cfg: dict, seed: int):
    """AdamW + parada temprana con la pérdida de validación. Devuelve el mejor estado y la curva de aprendizaje."""
    set_seed(seed)
    torch.set_num_threads(cfg["train"]["threads"])
    t = {**cfg["train"], **{k: hp[k] for k in ("lr", "loss") if k in hp}}
    model = construir(arch, hp, ctx)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    lv = torch.tensor(ctx["levels"])
    tr, va = tensores(D["train"]), tensores(D["val"])
    g = torch.Generator().manual_seed(seed)
    best, wait, state, hist = float("inf"), 0, None, []
    for _ in range(t["max_epochs"]):
        model.train()
        idx, total = torch.randperm(len(tr[2]), generator=g), 0.0
        for i in range(0, len(idx), t["batch_size"]):
            b = idx[i:i + t["batch_size"]]
            loss = perdida(*model(tr[0][b], tr[1][b], tr[2][b]), tr[3][b], lv, t["loss"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vl = perdida(*model(*va[:3]), va[3], lv, t["loss"]).item()
        hist.append({"train_loss": total / len(idx), "val_loss": vl})
        if vl < best - 1e-5:
            best, wait, state = vl, 0, copy.deepcopy(model.state_dict())
        elif (wait := wait + 1) >= t["patience"]:
            break
    model.load_state_dict(state)
    return model, hist


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def buscar(arch: str, ctx: dict, cfg: dict) -> pd.DataFrame:
    """Búsqueda aleatoria sin reemplazo sobre la grilla del YAML; criterio: MASE de validación."""
    sp = cfg["search"][arch]
    grid = [dict(zip(sp, v)) for v in itertools.product(*sp.values())]
    random.Random(cfg["search"]["seed"]).shuffle(grid)
    filas, cache = [], {}
    for hp in grid[:cfg["search"]["budget"][arch]]:
        D = cache.setdefault(hp["window"], datos(ctx, cfg, hp["window"]))
        t = time.time()
        model, hist = ajustar(arch, hp, D, ctx, cfg, cfg["search"]["seed"])
        p, q = predecir(model, D["val"])
        met = E.metricas(largo(ctx, D["val"], p, q, arch), ctx["denom"])
        filas.append({"model": arch, "hp": json.dumps(hp), "val_mase": met["mase"], "val_mae": met["mae"],
                      "val_pinball90": met["pinball90"], "val_cobertura90": met["cobertura90"],
                      "params": n_params(model), "epochs": len(hist), "seconds": time.time() - t})
        print(f"[search {arch}] {filas[-1]['hp']} val_mase={met['mase']:.4f} ({filas[-1]['seconds']:.0f}s)", flush=True)
    return pd.DataFrame(filas)


def guardar_artefacto(arch, hp, model, hist, ctx, cfg, seed, val_met):
    torch.save({"arch": arch, "hp": hp, "state": model.state_dict(), "rutas": ctx["rutas"],
                "scale": [float(x) for x in ctx["scale"]], "levels": ctx["levels"], "horizon": ctx["H"],
                "cal_cols": CAL_COLS, "seed": seed, "model_version": cfg["inference"]["model_version"],
                "splits": cfg["splits"], "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "val_metrics": {k: float(v) for k, v in val_met.items()}, "history": hist},
               FORECASTING_ARTIFACTS / f"{arch}.pt")


def semillas(arch: str, hp: dict, ctx: dict, cfg: dict, seeds: list[int]) -> tuple[pd.DataFrame, dict]:
    """Configuración congelada × N semillas: métricas val/test por semilla. Guarda el artefacto de la semilla de despliegue."""
    D = datos(ctx, cfg, hp["window"])
    filas, deploy = [], {}
    for seed in seeds:
        t = time.time()
        model, hist = ajustar(arch, hp, D, ctx, cfg, seed)
        secs = time.time() - t
        lv_ = {s: largo(ctx, D[s], *predecir(model, D[s]), arch) for s in ("val", "test")}
        mv, mt = (E.metricas(lv_[s], ctx["denom"]) for s in ("val", "test"))
        filas.append({"model": arch, "seed": seed, "runtime_s": secs, "parameters": n_params(model), "epochs": len(hist),
                      **{f"val_{k}": v for k, v in mv.items() if k != "n"}, **{f"test_{k}": v for k, v in mt.items() if k != "n"}})
        print(f"[seeds {arch}] seed={seed} val_mase={mv['mase']:.4f} test_mase={mt['mase']:.4f} ({secs:.0f}s)", flush=True)
        if seed == cfg["seeds"]["deploy"]:
            guardar_artefacto(arch, hp, model, hist, ctx, cfg, seed, mv)
            deploy = lv_
    return pd.DataFrame(filas), deploy


def resumen_semillas(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(("val_", "test_"))] + ["runtime_s", "parameters", "epochs"]
    return df.groupby("model")[cols].agg(["mean", "std", "min", "max"]).T
