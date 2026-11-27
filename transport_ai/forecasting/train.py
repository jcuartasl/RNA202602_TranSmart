"""Entrenamiento de modelos, búsqueda de hiperparámetros y 30 semillas."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from transport_ai.common.paths import FORECASTING_ARTIFACTS, EXPERIMENTS, CONFIGS
from transport_ai.common.logging_utils import get_logger, setup_logging, log_metrics
from transport_ai.common.serialization import save_checkpoint, save_json
from transport_ai.common.seeds import set_seed, SEEDS, DEPLOY_SEED
from transport_ai.forecasting.models import (
    create_model,
    combined_loss,
    poisson_nll_loss,
    pinball_loss_torch,
)

logger = get_logger(__name__)


def cargar_config() -> dict:
    """Carga configuración forecast.yaml."""
    with open(CONFIGS / "forecast.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def preparar_datos_torch(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler = None,
    fit_scaler: bool = False,
) -> tuple[DataLoader, StandardScaler]:
    """
    Prepara DataLoader de PyTorch con escalado opcional.

    Args:
        X: (n_samples, window_size, n_features)
        y: (n_samples, horizon)
        indices: Índices a usar
        scaler: StandardScaler ya ajustado (o None)
        fit_scaler: Si True, ajusta scaler en estos datos

    Returns:
        (DataLoader, scaler)
    """
    X_sub = X[indices]
    y_sub = y[indices]

    # Escalar features por muestra (flatten temporal)
    n_samples, window_size, n_features = X_sub.shape
    X_flat = X_sub.reshape(n_samples, -1)  # (n_samples, window_size * n_features)

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_flat)
    elif scaler is not None:
        X_scaled = scaler.transform(X_flat)
    else:
        X_scaled = X_flat

    # Reconstruir forma temporal para GRU/CNN1D
    X_final = X_scaled.reshape(n_samples, window_size, n_features)

    # Tensors
    X_tensor = torch.FloatTensor(X_final)
    y_tensor = torch.FloatTensor(y_sub)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    return loader, scaler


def entrenar_epoca(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: str = "cpu",
    grad_clip: float = 1.0,
) -> float:
    """Entrena una época."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward
        if isinstance(model, nn.Module) and hasattr(model, 'gru') or hasattr(model, 'conv'):
            # GRU/CNN1D: output directo
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
        else:
            # MLP: output directo
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def validar(
    model: nn.Module,
    loader: DataLoader,
    loss_fn,
    device: str = "cpu",
) -> float:
    """Evalúa en validación."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


def entrenar_modelo(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """
    Entrena modelo con early stopping.

    Returns:
        (best_model, history)
    """
    # Configurar pérdida
    loss_type = config.get("loss", "poisson_nll")
    quantile = config.get("quantile", 0.90)
    pinball_weight = config.get("pinball_weight", 0.5)

    def loss_fn(y_pred, y_true):
        # Modelos actuales son single-head (solo pronóstico puntual).
        # Para P90 real se necesitaría dos cabezas o pérdida cuantílica directa.
        # Aquí usamos pérdida puntual; P90 se estima en inferencia con factor heurístico.
        if loss_type == "poisson_nll":
            return poisson_nll_loss(y_pred, y_true)
        elif loss_type == "pinball":
            return pinball_loss_torch(y_pred, y_true, quantile)
        elif loss_type == "mse":
            return nn.MSELoss()(y_pred, y_true)
        elif loss_type == "mae":
            return nn.L1Loss()(y_pred, y_true)
        elif loss_type == "combined":
            # Combined requiere modelo con dos cabezas (point + quantile).
            # No implementado en modelos actuales; usar poisson_nll.
            logger.warning("Loss 'combined' requiere dos cabezas; usando poisson_nll")
            return poisson_nll_loss(y_pred, y_true)
        else:
            raise ValueError(f"Loss no soportado: {loss_type}")

    # Optimizador
    opt_config = config.get("training", {})
    lr = opt_config.get("learning_rate", 0.001)
    weight_decay = opt_config.get("weight_decay", 1e-4)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Early stopping
    patience = opt_config.get("early_stopping_patience", 10)
    min_delta = opt_config.get("early_stopping_min_delta", 1e-4)
    max_epochs = opt_config.get("max_epochs", 100)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    model.to(device)

    for epoch in range(max_epochs):
        train_loss = entrenar_epoca(model, train_loader, optimizer, loss_fn, device)
        val_loss = validar(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        log_metrics(logger, f"epoch {epoch+1}/{max_epochs}",
                   {"train_loss": train_loss, "val_loss": val_loss})

        # Early stopping
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping en época {epoch+1}")
                break

    # Restaurar mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def entrenar_semilla(
    seed: int,
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    model_type: str,
    model_config: dict,
    device: str = "cpu",
) -> dict:
    """
    Entrena un modelo con una semilla específica.

    Returns:
        Diccionario con métricas, history, tiempo, parámetros
    """
    set_seed(seed)

    # Preparar datos
    train_loader, scaler = preparar_datos_torch(X, y, splits["train"], fit_scaler=True)
    val_loader, _ = preparar_datos_torch(X, y, splits["val"], scaler=scaler)

    # Crear modelo
    model = create_model(model_type, model_config)

    # Entrenar
    start_time = time.time()
    model, history = entrenar_modelo(model, train_loader, val_loader, model_config, device)
    train_time = time.time() - start_time

    # Evaluar en test
    test_loader, _ = preparar_datos_torch(X, y, splits["test"], scaler=scaler)

    model.eval()
    model.to(device)
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds.append(y_pred.cpu().numpy())
            all_true.append(y_batch.numpy())

    y_pred_test = np.vstack(all_preds)
    y_true_test = np.vstack(all_true)

    # Métricas (usar point forecast, quantile = point * 1.2 para baseline)
    from transport_ai.common.metrics import evaluate_forecast
    # Necesitamos y_train para MASE - obtener de splits train
    _, scaler_train = preparar_datos_torch(X, y, splits["train"], fit_scaler=True)
    # Para MASE necesitamos la serie original de entrenamiento - simplificación
    y_train_flat = y[splits["train"]].mean(axis=1)  # Aproximación

    # Métricas por horizonte
    metrics = {}
    for i in range(y_pred_test.shape[1]):
        y_true_i = y_true_test[:, i]
        y_pred_i = y_pred_test[:, i]
        metrics[f"day_{i+1}"] = {
            "mae": float(np.mean(np.abs(y_true_i - y_pred_i))),
            "rmse": float(np.sqrt(np.mean((y_true_i - y_pred_i) ** 2))),
        }

    # Métricas agregadas
    agg_metrics = {
        "mae": float(np.mean(np.abs(y_true_test - y_pred_test))),
        "rmse": float(np.sqrt(np.mean((y_true_test - y_pred_test) ** 2))),
        "train_time_sec": train_time,
        "n_params": model.get_num_params(),
    }

    return {
        "seed": seed,
        "metrics": agg_metrics,
        "metrics_by_day": metrics,
        "history": history,
        "model_state": {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
    }


def entrenar_30_semillas(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    model_type: str,
    model_config: dict,
    seeds: list[int] = None,
    deploy_seed: int = DEPLOY_SEED,
    device: str = "cpu",
) -> dict:
    """
    Entrena modelo con 30 semillas independientes.

    Args:
        X, y: Datos completos
        splits: Dict con índices train/val/test
        model_type: "mlp" o "gru"
        model_config: Configuración del modelo
        seeds: Lista de semillas (default: SEEDS)
        deploy_seed: Semilla para modelo de despliegue (pre-definida)

    Returns:
        Resultados completos con mejor modelo para despliegue
    """
    if seeds is None:
        seeds = SEEDS

    logger.info(f"Entrenando {model_type} con {len(seeds)} semillas...")

    resultados = []
    best_val_loss = float("inf")
    best_seed = None
    best_state = None
    best_scaler = None

    for seed in seeds:
        logger.info(f"  Semilla {seed}...")
        resultado = entrenar_semilla(seed, X, y, splits, model_type, model_config, device)
        resultados.append(resultado)

        # Track mejor por validación (usar último val_loss del history)
        val_loss = resultado["history"]["val_loss"][-1]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_seed = seed
            best_state = resultado["model_state"]
            best_scaler = resultado["scaler"]

    # Re-entrenar con deploy_seed en train+val combinado para modelo final
    logger.info(f"Re-entrenando modelo final con deploy_seed={deploy_seed} en train+val...")
    set_seed(deploy_seed)

    # Combinar train+val
    trainval_indices = np.concatenate([splits["train"], splits["val"]])
    trainval_loader, final_scaler = preparar_datos_torch(X, y, trainval_indices, fit_scaler=True)

    final_model = create_model(model_type, model_config)
    final_model, final_history = entrenar_modelo(
        final_model, trainval_loader, trainval_loader,  # Usar mismo para val (early stopping interno)
        model_config, device
    )

    # Guardar artefacto de despliegue
    artifact_dir = FORECASTING_ARTIFACTS / model_type
    artifact_dir.mkdir(parents=True, exist_ok=True)

    save_checkpoint(
        artifact_dir / "best.pt",
        final_model,
        config=model_config,
        scaler=final_scaler,
        metadata={
            "model_type": model_type,
            "deploy_seed": deploy_seed,
            "best_val_seed": best_seed,
            "best_val_loss": best_val_loss,
            "n_params": final_model.get_num_params(),
        },
        version="1.0.0",
    )

    # Guardar resultados de 30 semillas
    seeds_results = []
    for r in resultados:
        seeds_results.append({
            "seed": r["seed"],
            "mae": r["metrics"]["mae"],
            "rmse": r["metrics"]["rmse"],
            "train_time_sec": r["metrics"]["train_time_sec"],
            "n_params": r["metrics"]["n_params"],
            "val_loss_final": r["history"]["val_loss"][-1],
        })

    save_json(seeds_results, artifact_dir / "seeds_results.json")

    # Resumen estadístico
    maes = [r["mae"] for r in seeds_results]
    rmses = [r["rmse"] for r in seeds_results]

    summary = {
        "model_type": model_type,
        "n_seeds": len(seeds),
        "deploy_seed": deploy_seed,
        "best_val_seed": best_seed,
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes, ddof=1)),
        "mae_min": float(np.min(maes)),
        "mae_max": float(np.max(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses, ddof=1)),
        "rmse_min": float(np.min(rmses)),
        "rmse_max": float(np.max(rmses)),
        "n_params": final_model.get_num_params(),
    }

    save_json(summary, artifact_dir / "summary.json")
    logger.info(f"Resumen {model_type}: MAE={summary['mae_mean']:.2f}±{summary['mae_std']:.2f}, RMSE={summary['rmse_mean']:.2f}±{summary['rmse_std']:.2f}")

    return {
        "summary": summary,
        "seeds_results": seeds_results,
        "final_model_path": str(artifact_dir / "best.pt"),
        "scaler": final_scaler,
    }


def buscar_hiperparametros(
    X: np.ndarray,
    y: np.ndarray,
    splits: dict,
    model_type: str,
    search_space: dict,
    n_trials: int = 30,
    device: str = "cpu",
) -> dict:
    """
    Búsqueda aleatoria de hiperparámetros.

    Returns:
        Mejor configuración encontrada
    """
    import random

    logger.info(f"Búsqueda hiperparámetros {model_type}: {n_trials} trials")

    best_config = None
    best_val_loss = float("inf")
    all_trials = []

    base_config = cargar_config()["models"][model_type]

    for trial in range(n_trials):
        # Samplear configuración
        config = base_config.copy()
        config["model_kwargs"] = {}

        for param, values in search_space.items():
            config["model_kwargs"][param] = random.choice(values)

        # Entrenar con una semilla (rápido)
        set_seed(42)
        train_loader, scaler = preparar_datos_torch(X, y, splits["train"], fit_scaler=True)
        val_loader, _ = preparar_datos_torch(X, y, splits["val"], scaler=scaler)

        model = create_model(model_type, config["model_kwargs"])
        model, history = entrenar_modelo(model, train_loader, val_loader, config, device)

        val_loss = history["val_loss"][-1]

        trial_result = {
            "trial": trial,
            "config": config["model_kwargs"],
            "val_loss": val_loss,
            "epochs": len(history["val_loss"]),
        }
        all_trials.append(trial_result)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = config["model_kwargs"].copy()
            logger.info(f"  Trial {trial}: NUEVO MEJOR val_loss={val_loss:.4f} config={best_config}")

    # Guardar resultados de búsqueda
    save_json({
        "best_config": best_config,
        "best_val_loss": best_val_loss,
        "all_trials": all_trials,
    }, EXPERIMENTS / "m1" / f"hparam_search_{model_type}.json")

    return best_config