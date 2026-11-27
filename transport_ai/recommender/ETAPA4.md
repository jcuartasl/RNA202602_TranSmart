# P4 · Etapa 4: auditoría de entidades, análisis comercial y figuras

Este paquete **no cambia los splits, no reentrena y no sobrescribe pesos**. Ejecutarlo desde la raíz de `RNA202602_TranSmart` con el entorno `.venv_p4` activo, sobre los artefactos generados en etapas 1–3.

## 1. Dependencias

```powershell
python -m pip install numpy matplotlib
python -m unittest discover -s tests/unit -p 'test_recommender_p4*.py' -v
```

El warning `Failed to initialize NumPy` se debe a la ausencia de NumPy; las corridas previas que terminaron bien **no necesitan repetirse por ese warning**. Coordinar versiones exactas con P5 antes de cambiar `requirements.txt` global.

## 2. Auditoría de posibles duplicados semánticos

```powershell
python -m transport_ai.recommender.cli audit-catalog
```

Guarda `experiments/m3/diagnostics/m3_catalog_audit.json` con las agrupaciones de `Name`, `Name+State` y `Name+State+Type` normalizadas; imprime conteos. **No fusionar ni eliminar IDs** solo por nombre: revisar ejemplos y discutir con el equipo si son realmente el mismo destino. Si se altera la unidad de análisis, se debe reconstruir el protocolo completo y aplicarlo por igual a todos los modelos, nunca seleccionar un cambio por su mejora de test.

## 3. Métricas por K y ejemplos de 5 usuarios (validación)

```powershell
python -m transport_ai.recommender.cli report-p4 --split validation --ks 1 3 5 10 20
```

Resultados:
- `experiments/m3/diagnostics/m3_metrics_vs_k_validation.csv`: Precision, Recall, NDCG, cobertura, HHI, top-5 share y segmentación pocos/muchos por modelo y K.
- `experiments/m3/diagnostics/m3_report_validation.json`: detalles por usuario, cinco perfiles, historia exclusivamente disponible en train, recomendaciones y distribuciones de train.

Comparación de los cuatro enfoques: popularidad, kNN, contenido y red de dos torres. Todos comparten candidato, split y regla de exclusión de interacciones conocidas; se usa el `best.pt` ya elegido con validación.

## 4. Figuras reproducibles

```powershell
python -m transport_ai.recommender.cli figures-p4 --split validation
```

Guarda imágenes `reports/figures/m3_*.png`:
1. Cola larga de interacciones positivas por usuario/destino.
2. Precision, Recall y NDCG en función de K para los cuatro modelos.
3. Cobertura y HHI vs K.
4. PCA 2D de las representaciones de destinos **vistos en train**, con color por categoría. PCA es una proyección descriptiva, no prueba grupos naturales ni causalidad.
5. Curvas de BCE **train** y NDCG **validación** por época. Nota: **no son pérdida train/val equivalentes**. Sigue pendiente establecer y producir una pérdida de validación comparable sin fuga.
6. Boxplot de semillas; hiperparámetros; diagrama de arquitectura dos torres.

## 5. Campaña hipotética (sin fingir conversiones observadas)

```powershell
python -m transport_ai.recommender.cli campaign-p4 --audience-type beach --exposure-rate 0.50 --conversion-rate 0.03
```

Las conversiones esperadas son `usuarios_elegibles × tasa_exposición × tasa_conversión_supuesta`. Ninguna tasa se ha calibrado a compras reales; no tratar la sigmoid del recomendador como conversión. El destino de ejemplo se selecciona por interacciones positivas de train dentro del tipo elegido.

## 6. Evaluación final sin ajuste posterior (solo cuando se cierre el diseño)

Los 30 checkpoints ya están entrenados. Si el equipo decide congelar el protocolo, correr **una única vez** y documentar:

```powershell
python -m transport_ai.recommender.cli report-p4 --split test --ks 1 3 5 10 20
python -m transport_ai.recommender.cli report-p4 --split cold_test --ks 1 3 5 10 20
```

IMPORTANTE: el baseline inicial ya mostró test en la etapa 1, así que **el test actual no está completamente sellado**. No elegir parámetros, modelos ni limpieza mirando ese resultado. Registrar explícitamente la observación previa y discutir con P5/profesor si el conjunto de prueba debe quedar como validación exploratoria o requieren un diseño verdaderamente sellado.

## Requisitos aún por completar

- Verificar en la documentación original de Kaggle si los datos son reales o sintéticos (no asumir).
- Revisar duplicación de destinos con ID distinto; definir semántica de item y cuantificar impacto.
- Pérdida de validación comparable y clara, con parada temprana señalada.
- Derivaciones BCE/embeddings/factorización de matrices y explicación de métricas.
- Documentar hipótesis, análisis de sesgos, política cold-start, campaña y limitaciones.
- Prueba de integración con P5: `recommend(user_id=None, profile=None, k=5)` sin entrenar y con artefactos empaquetados.
- Git clean clone, versions fijadas, commits, fuentes y cacería de alucinaciones.
