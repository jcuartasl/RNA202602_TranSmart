# P4 — Etapa 2: referencias clásicas, preferencias e interfaz preliminar

**Estado:** se implementó un filtrado colaborativo clásico ítem-ítem (coseno de co-ocurrencias de `train`) y dos políticas de recomendación para nuevos usuarios: popularidad de entrenamiento y coincidencia entre `Preferences` y `Type`. No se afirma que esto sea la red neuronal final.

## Comandos, desde la raíz del repo

```powershell
python -m unittest discover -s tests/unit -p 'test_recommender_p4*.py' -v
python -m transport_ai.recommender.cli fit-classical --neighbors 20
python -m transport_ai.recommender.cli compare --split validation --k 5
python -m transport_ai.recommender.cli recommend --preferences 'Beaches, Historical' --method content --k 5
```

`compare` permite `--split test` y `--split cold_test`, **pero no deben usarse para ajustar hiperparámetros**. Elegir `neighbors` y `k` según validación; reservar test hasta terminar diseños, excepto si ya se consultó (en ese caso, documentar que el sellado original quedó comprometido). Para cold-start, comparar en usuarios retenidos solo cuando se haya congelado la política.

`fit-classical` genera `artifacts/recommender/item_knn.json`: el programa `recommend(...)` lo carga para inferir sin volver a entrenar. No usa la columna `Popularity` de Kaggle por procedencia incierta. La similitud colaborativa puede ser cero si ningún usuario de entrenamiento tiene dos destinos positivos, lo que no es un error de programación. En tal caso aplica respaldo de popularidad.

## Criterios y advertencias

* **Split**: `train`, `validation`, `test` y `cold_test` permanecen iguales para los tres baselines. Los positivos son calificaciones >=4 (supuesto experimental) y las observaciones de 1–3 no se convierten en negativos observados de clic.
* **Separación**: `train` solo alimenta conteos y similitudes; `Preferences` es información disponible en el perfil desde antes de una interacción de prueba. No se usan reseñas ni su texto, nombre ni correo.
* **Ranking**: se excluyen destinos observados de entrenamiento, se usa el mismo catálogo de 1.000 destinos como candidatos, con empates por ID. Cobertura se calcula sobre el catálogo completo.
* **Desempeño**: el conjunto es muy disperso; resultados cero o bajos pueden ser coherentes. Reportar honestamente el tamaño de cada segmento.
* **Limitación importante**: los ítems de test no necesariamente fueron vistos en train; se informa cobertura del catálogo y deben añadirse análisis de candidatos fríos. Los modelos colaborativos no pueden aprender factores de un ítem jamás visto.
* **API preliminar**: `transport_ai.recommender.infer.recommend(user_id=None, profile=None, k=5, method='content')`. Devuelve lista de diccionarios `destination_id`, `destination_name`, `score`, `reason`, `history`. El historial contiene solo positivos observados de entrenamiento.
* **Pendientes**: modelo neuronal con contenido real, búsqueda de hiperparámetros, 30 semillas, figuras, informe matemático, análisis comercial y artefacto neuronal final. No confundir baselines con entrega completa.
