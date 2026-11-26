# P4 — Recomendador de destinos internacionales (TranSmart)

El módulo trabaja con **Travel Recommendation Dataset, Aman Mehra, Kaggle** (fuente exigida por el profesor). Es un experimento académico; la procedencia real/sintética **no está verificada en la documentación original**, así que no se debe inferir realismo comercial de los resultados.

## Dos experimentos separados

1. **Experimento histórico por ID**: 1.000 IDs, 999 usuarios y 1.998 pares usuario–ID, con métricas muy bajas (densidad 0,2 %). Se conserva para comparar el efecto de los datos repetidos.
2. **Experimento normalizado recomendado para la app**: agrupar por nombre+estado+tipo normalizados → **cinco destinos lógicos**. Se eliminan del aprendizaje 160 pares usuario–lugar con calificaciones contradictorias. **No reutilizar checkpoints del experimento original**.

## Entradas y salida

`transport_ai.recommender.infer.recommend(user_id=None, profile={"preferences": "Beaches, Historical"}, k=3, method="neural", processed=..., neural_artifact=...)` devuelve una lista ordenada de `destination_id`, `destination_name`, `score`, `reason`, `history` sin IDs duplicados, sin entrenamiento. Para un usuario existente proporcionar `user_id` en lugar de perfil nuevo.

La salida numérica de la red es **score de ranking**, no probabilidad calibrada de compra. El catálogo y el historial se cargan desde archivos procesados; no hay llamadas a Kaggle en inferencia.

## Evaluación (catálogo normalizado)

Usar `validation` para comparar y seleccionar hiperparámetros. La mejor configuración se entrena con 30 semillas independientes y el modelo preferido se selecciona únicamente en validación. Reportar la media y desviación entre semillas, además de la mejor corrida, y reconocer que el conjunto de validación tiene 17 usuarios. Se consulta `cold_test` solo con modelos congelados; no volver a ajustar el modelo después de inspeccionarlo.

```powershell
python -m unittest discover -s tests/unit -p "test_recommender_p4*.py" -v
python -m transport_ai.recommender.cli compare --output data/processed/recommender_entities --artifact artifacts/recommender_entities/item_knn.json --split validation --k 1
python -m transport_ai.recommender.cli eval-neural --output data/processed/recommender_entities --results experiments/m3/neural_entities --neural-artifacts artifacts/recommender_entities/neural --split validation --k 1
python -m transport_ai.recommender.cli report-p4 --output data/processed/recommender_entities --artifact artifacts/recommender_entities/item_knn.json --neural-artifacts artifacts/recommender_entities/neural --report-dir experiments/m3/diagnostics_entities --split validation --ks 1 2 3
python -m transport_ai.recommender.cli figures-p4 --output data/processed/recommender_entities --neural-artifacts artifacts/recommender_entities/neural --results experiments/m3/neural_entities --report-dir experiments/m3/diagnostics_entities --figure-dir reports/figures/m3_entities --split validation
```

**Para entrega final:** `report-p4 --split cold_test` y `eval-neural --split cold_test` sobre la variante normalizada, sin selección adicional; probar también test warm al cerrar metodología y conservar el informe de limitaciones (en el proyecto ya se observó el test original por ID).

## Documentación

- [ENTREGA_P5.md](ENTREGA_P5.md): contrato y rutas para Streamlit.
- [reports/blog_p4_borrador.md](../../reports/blog_p4_borrador.md): fundamentación, resultados de validación y limitaciones.
- `experiments/m3/neural_entities`: búsqueda, 30 semillas y resúmenes de validación corregidos.
- `reports/figures/m3_entities`: figuras corregidas; **jamás mezclar estas con las curvas por ID original**.

Falta abordar con rigor el requisito de **pérdida de validación**. El entrenamiento guarda BCE(train) y NDCG(validation), que no son la misma pérdida. Obtener BCE(validation) requeriría reservar también no-positivos explícitos del entrenamiento y rehacer partición/entrenamientos. No renombrar NDCG como pérdida.

No subir al repositorio datos crudos con nombres y emails, tokens, entorno virtual, ni modelos de semillas innecesarios para la app. Revisar licencia antes de redistribuir datos procesados; coordinar con P5 versiones exactas de PyTorch/NumPy/Matplotlib en `requirements.txt`.
