# P4 â€” contrato de integraciÃ³n y entrega a P5

**Unidad publicada:** destino lÃ³gico (`place_0001` â€¦ `place_0005`), no los 1000 IDs crudos repetidos.
Los modelos de la versiÃ³n de IDs originales son un **experimento separado** y no se deben conectar a la app final si se adopta la versiÃ³n por entidad.

## Llamada en la aplicaciÃ³n Streamlit

Desde la raÃ­z del repositorio (el directorio que contiene `transport_ai/`):

```python
from pathlib import Path
from transport_ai.recommender.infer import recommend

recommendations = recommend(
    user_id=None,  # O un UserID existente para mostrar su historial de entrenamiento
    profile={"preferences": "Beaches, Historical"},  # onboarding de un usuario nuevo
    k=3,
    method="neural",
    processed=Path("data/processed/recommender_entities_v2"),
    neural_artifact=Path("artifacts/recommender_entities_v2/neural/best.pt"),
)
```

La salida es una lista ordenada de diccionarios con `destination_id`, `destination_name`, `score`, `reason` y `history` (vacÃ­o para un usuario nuevo). No duplica `destination_id`, no entrena, no llama APIs externas. Con `user_id` conocido, pasar el ID como texto; `profile=None` usa preferencias registradas en el archivo procesado.

**Importante:** `score` es una salida del modelo para ordenar, **no** una probabilidad calibrada de compra; mostrar como puntuaciÃ³n, no como Â«X % comprarÃ¡Â».

## Artefactos que P5 necesita en el clon limpio

- Para **inferencia neuronal** bastan `data/processed/recommender_entities_v2/{catalog,profiles,split_manifest}.csv`; `interactions.csv` se necesita para **reconstruir los experimentos**, no para iniciar inferencia. Confirmar autorizaciÃ³n de redistribuciÃ³n antes de publicarlos.
- `artifacts/recommender_entities_v2/neural/best.pt` para inferencia neuronal.
- `artifacts/recommender_entities_v2/item_knn.json` si se expone tambiÃ©n el baseline.
- `transport_ai/recommender/` y las dependencias Python de versiÃ³n fija coordinadas por P5.

No descargar ni entrenar durante el inicio de Streamlit. P5 debe envolver la carga del artefacto en cachÃ© (opcional) y presentar mensajes claros ante archivos faltantes.

## Comandos de verificaciÃ³n local

```powershell
python -m unittest discover -s tests/unit -p "test_recommender_p4*.py" -v
python -m transport_ai.recommender.cli recommend --method neural --output data/processed/recommender_entities_v2 --neural-artifacts artifacts/recommender_entities_v2/neural --preferences "Beaches, Historical" --k 3
```

Los nombres de preferencias disponibles son categorÃ­as `Beach`, `Historical`, `Nature`, `Adventure`, `City`; sinÃ³nimos reconocidos en `coldstart.py`. El selector de P5 debe enviar estas categorÃ­as a `profile["preferences"]`.

## Riesgos y lÃ­mites visibles en la interfaz

- Solo hay cinco destinos lÃ³gicos tras la normalizaciÃ³n `name+state+type`; el experimento no prueba que sean lugares distintos o iguales mediante geolocalizaciÃ³n.
- El cliente Â«nuevoÂ» tiene preferencias pero no historial; no se debe mostrarle ninguna valoraciÃ³n de prueba como historia.
- Los datos de Kaggle pueden no reflejar usuarios/compras reales; **procedencia real/sintÃ©tica no verificada**.
- No usar resultados de India para planear automÃ¡ticamente flota de rutas colombianas.

## Dependencias y opciÃ³n hÃ­brida (actualizaciÃ³n)

- Entorno probado: Python 3.13.9 / torch 2.15.0+cpu / numpy 2.5.5 / matplotlib 3.11.2.
- Archivo `requirements-p4.txt` en la raÃ­z, sin alterar el `requirements.txt` del equipo.
- `method="neural"`: ranking neuronal original. `method="neural-hybrid"` y `preference_boost=0.02`: priorizaciÃ³n **experimental** de coincidencias de tipo, no entrenamiento nuevo.
- ValidaciÃ³n preliminar del hÃ­brido (17 usuarios): NDCG@1 cae 0.7059 â†’ 0.5882, afinidad sube 23.53% â†’ 41.18%; **no usar como mejora de precisiÃ³n**. Los resultados corresponden al modelo anterior al nuevo holdout de no-positivos.

## ValidaciÃ³n BCE y metodologÃ­a de entrenamiento

El experimento inicial no tenÃ­a BCE(validation). Esta revisiÃ³n reserva en `split_manifest.csv` una clase `validation_nonpositive` (calificaciones observadas 1â€“3), excluida del entrenamiento, para calcular BCEWithLogitsLoss sobre **positivos y no-positivos explÃ­citos retenidos** en cada Ã©poca. El NDCG de validaciÃ³n se sigue calculando Ãºnicamente sobre positivos no vistos. No tratar ausencia como 0.

**No reutilizar el checkpoint previo con este split:** ejecutar `prepare-entities`, `tune-neural` y `train-neural` en **rutas _v2 separadas**, regenerar figuras y reportar de nuevo los nÃºmeros. La figura adicional `m3_10_curvas_bce_train_validation.png` muestra train y validaciÃ³n de la semilla seleccionada. La prueba original de clon limpio sigue siendo una verificaciÃ³n vÃ¡lida del artefacto anterior, no del V2 hasta volver a probarlo.

