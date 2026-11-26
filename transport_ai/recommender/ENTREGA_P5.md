# P4 — contrato de integración y entrega a P5

**Unidad publicada:** destino lógico (`place_0001` … `place_0005`), no los 1000 IDs crudos repetidos.
Los modelos de la versión de IDs originales son un **experimento separado** y no se deben conectar a la app final si se adopta la versión por entidad.

## Llamada en la aplicación Streamlit

Desde la raíz del repositorio (el directorio que contiene `transport_ai/`):

```python
from pathlib import Path
from transport_ai.recommender.infer import recommend

recommendations = recommend(
    user_id=None,  # O un UserID existente para mostrar su historial de entrenamiento
    profile={"preferences": "Beaches, Historical"},  # onboarding de un usuario nuevo
    k=3,
    method="neural",
    processed=Path("data/processed/recommender_entities"),
    neural_artifact=Path("artifacts/recommender_entities/neural/best.pt"),
)
```

La salida es una lista ordenada de diccionarios con `destination_id`, `destination_name`, `score`, `reason` y `history` (vacío para un usuario nuevo). No duplica `destination_id`, no entrena, no llama APIs externas. Con `user_id` conocido, pasar el ID como texto; `profile=None` usa preferencias registradas en el archivo procesado.

**Importante:** `score` es una salida del modelo para ordenar, **no** una probabilidad calibrada de compra; mostrar como puntuación, no como «X % comprará».

## Artefactos que P5 necesita en el clon limpio

- `data/processed/recommender_entities/{catalog,profiles,interactions,split_manifest}.csv` y archivos auxiliares de partición generados por `prepare-entities`, si su redistribución está autorizada.
- `artifacts/recommender_entities/neural/best.pt` para inferencia neuronal.
- `artifacts/recommender_entities/item_knn.json` si se expone también el baseline.
- `transport_ai/recommender/` y las dependencias Python de versión fija coordinadas por P5.

No descargar ni entrenar durante el inicio de Streamlit. P5 debe envolver la carga del artefacto en caché (opcional) y presentar mensajes claros ante archivos faltantes.

## Comandos de verificación local

```powershell
python -m unittest discover -s tests/unit -p "test_recommender_p4*.py" -v
python -m transport_ai.recommender.cli recommend --method neural --output data/processed/recommender_entities --neural-artifacts artifacts/recommender_entities/neural --preferences "Beaches, Historical" --k 3
```

Los nombres de preferencias disponibles son categorías `Beach`, `Historical`, `Nature`, `Adventure`, `City`; sinónimos reconocidos en `coldstart.py`. El selector de P5 debe enviar estas categorías a `profile["preferences"]`.

## Riesgos y límites visibles en la interfaz

- Solo hay cinco destinos lógicos tras la normalización `name+state+type`; el experimento no prueba que sean lugares distintos o iguales mediante geolocalización.
- El cliente «nuevo» tiene preferencias pero no historial; no se debe mostrarle ninguna valoración de prueba como historia.
- Los datos de Kaggle pueden no reflejar usuarios/compras reales; **procedencia real/sintética no verificada**.
- No usar resultados de India para planear automáticamente flota de rutas colombianas.
