# Etapa 5 — Calidad de identidad de destinos y evaluación corregida

## Hallazgo experimental (dataset de la asignatura)

La auditoría reportó **1.000 IDs diferentes**, pero solo **cinco combinaciones distinguibles** de nombre + estado + tipo. Los NDCG@5, la cobertura y las recomendaciones por `DestinationID` de las etapas 1–4 **no pueden interpretarse como si representaran 1.000 destinos físicos distintos**. La salida con cuatro registros llamados «Taj Mahal» demuestra el problema visible en la interfaz.

## Principios

1. Conservar los artefactos originales y resultados como experimento **a nivel de ID**. No presentarlos como resultados de un catálogo de mil lugares distintos.
2. Hacer un análisis secundario **por entidad lógica**, sin modificar las entradas originales; documentar la regla de identidad *nombre + estado + tipo* y sus límites.
3. **Agrupar los IDs antes de dividir los pares usuario–destino**, para no separar el mismo lugar lógico entre train, validation y test para un mismo usuario. `build_split` excluye grupos con valoraciones explícitas contradictorias.
4. Reentrenar los tres modelos desde cero sobre las mismas particiones nuevas y reportar K=1,2,3. K=5 sobre un catálogo de solo cinco lugares es poco informativo y puede recomendar todos los disponibles.
5. El origen real/sintético de los datos **no ha sido verificado** en la ficha del autor de Kaggle; no declararlo sin evidencia verificable.
6. El test original ya se consultó en el experimento por ID. Documentar la exposición previa como limitación; reservar la nueva validación para selección y reportar test una vez al final, no iterar contra test.

## Comandos desde la raíz del monorrepositorio

Primero auditar el problema en los datos existentes:

```powershell
python -m transport_ai.recommender.cli audit-entities
```

Luego crear el segundo dataset procesado y nueva partición (no elimina el anterior):

```powershell
python -m transport_ai.recommender.cli prepare-entities
```

Entrenar clásicos bajo nueva granularidad:

```powershell
python -m transport_ai.recommender.cli fit-classical --output data/processed/recommender_entities --artifact artifacts/recommender_entities/item_knn.json --neighbors 3
python -m transport_ai.recommender.cli compare --output data/processed/recommender_entities --artifact artifacts/recommender_entities/item_knn.json --split validation --k 3
```

Entrenar neuronal con nueva granuralidad (se requieren pesos independientes):

```powershell
python -m transport_ai.recommender.cli tune-neural --output data/processed/recommender_entities --results experiments/m3/neural_entities --trials 3 --epochs 12 --k 3
python -m transport_ai.recommender.cli train-neural --output data/processed/recommender_entities --results experiments/m3/neural_entities --neural-artifacts artifacts/recommender_entities/neural --seeds 30 --seed-start 42 --epochs 16 --k 3
python -m transport_ai.recommender.cli eval-neural --output data/processed/recommender_entities --results experiments/m3/neural_entities --neural-artifacts artifacts/recommender_entities/neural --split validation --k 3
```

Gráficos y reporte en carpetas separadas para no sobrescribir la primera metodología:

```powershell
python -m transport_ai.recommender.cli report-p4 --output data/processed/recommender_entities --artifact artifacts/recommender_entities/item_knn.json --neural-artifacts artifacts/recommender_entities/neural --report-dir experiments/m3/entities --split validation --ks 1 2 3
python -m transport_ai.recommender.cli figures-p4 --output data/processed/recommender_entities --neural-artifacts artifacts/recommender_entities/neural --report-dir experiments/m3/entities --figure-dir reports/figures/m3_entities --split validation
python -m transport_ai.recommender.cli recommend --method neural --output data/processed/recommender_entities --neural-artifacts artifacts/recommender_entities/neural --preferences "Beaches, Historical" --k 3
```

**Para P5:** si se decide usar el catálogo normalizado, conectar `recommend(..., processed=Path('data/processed/recommender_entities'), neural_artifact=Path('artifacts/recommender_entities/neural/best.pt'))`. Mostrar K entre 1 y 3 preferentemente; informar que existen cinco lugares lógicos. Si un usuario ya conoce algunos, es normal que haya menos de K destinos nuevos disponibles.

## Pendientes para el informe académico

- Citar licencia y naturaleza del dataset desde su ficha original (pendiente).
- Añadir gráfica de **pérdida de validación** en vez de llamar pérdida a NDCG de validación. Para ello hay que definir negativos de validación conocidos y no recalibrar usando test.
- Ejecutar la comparación final una sola vez sobre el conjunto de prueba una vez congelados los modelos.
- Completar derivación matemática, discusión ética, APA y video individual.
- Registrar tiempos y commits, asegurar reproductibilidad con datos procesados y versiones fijadas junto a P5.
