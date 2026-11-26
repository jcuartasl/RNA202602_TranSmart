# P4 — Etapa 3: recomendador neuronal de dos torres

## Qué se implementó

Una **red neuronal real** con PyTorch: torre de usuario (`UserID` + vector multietiqueta `Preferences`) y torre de destino (`DestinationID` + vector `Type`), embeddings entrenables, proyecciones MLP, puntuación por producto interno y pérdida BCE con logits. Índice 0 representa usuarios o destinos sin historial entrenable; un usuario nuevo puede obtener un ranking con sus preferencias. La salida se integra a `recommend(..., method='neural')` para P5, sin reentrenar durante la consulta.

## Cómo se define el problema

`rating >= 4` se trata como positivo y `rating 1-3` como no positivo (supuesto, no implica que 1-3 sea necesariamente rechazo a visitar). La partición existente ya fija train/validation/test/cold_test. Se entrenan únicamente positivos `train` y calificaciones no positivas `observed_nonpositive`; **no se usan interacciones positivas retenidas**. El parámetro `negative_ratio` **submuestrea negativos explícitos** por época y no convierte destinos jamás observados en rechazos inventados. No es válido afirmar generalización temporal con reviews sin fecha.

El modelo trabaja con 1000 destinos candidatos; cada usuario excluye ítems observados de entrenamiento. La validación puntúa todos los candidatos restantes, no solamente 5 negativos fáciles. `Popularity` del catálogo no se utiliza, por posible procedencia agregada.

## Preparación e instalación

En el entorno virtual de P4, desde la raíz del repositorio:

```powershell
python -m pip show torch
# Si no existe:
python -m pip install torch
python -m unittest discover -s tests/unit -p 'test_recommender_p4*.py' -v
```

Acordar después con P5 la versión exacta para `requirements.txt` compartido (`python -m pip show torch`); nunca subir `venv` ni `__pycache__`.

## Comandos: búsqueda y entrenamiento

```powershell
# Ya se ejecutó `prepare` antes: no regenerar splits por diversión.
python -m transport_ai.recommender.cli tune-neural --trials 3 --epochs 12 --k 5
python -m transport_ai.recommender.cli train-neural --seeds 30 --seed-start 42 --epochs 16 --k 5
python -m transport_ai.recommender.cli eval-neural --split validation --k 5
# SOLO AL CONGELAR EL MODELO Y EL PROTOCOLO:
python -m transport_ai.recommender.cli eval-neural --split test --k 5
python -m transport_ai.recommender.cli eval-neural --split cold_test --k 5
python -m transport_ai.recommender.cli recommend --method neural --preferences 'Beaches, Historical' --k 5
```

Demo de entrenamiento ligero (no sustituye 30 semillas):

```powershell
python -m transport_ai.recommender.cli train-neural --seeds 1 --epochs 4 --k 5
```

**No utilizar la demo en el mismo directorio de artefactos que una corrida final de 30 semillas.** Si la necesitan, usen `--neural-artifacts artifacts/recommender/neural-demo --results experiments/m3/neural-demo` para no sobrescribir modelos finales.

## Salidas

- `experiments/m3/neural/search_results.json`: configuraciones candidatas y NDCG de validación.
- `experiments/m3/neural/best_config.json`: configuración fijada sin observar test.
- `artifacts/recommender/neural/seed_*.pt`: 30 checkpoints independientes; `best.pt` seleccionado por validación.
- `experiments/m3/neural/neural_validation_runs.json`: pérdidas y curva NDCG por época; tiempos y parámetros entrenables.
- `experiments/m3/neural/neural_validation_summary.json`: media, desviación estándar, mejor y peor.
- `experiments/m3/neural/neural_test_results.json` y `neural_cold_test_results.json`: evaluación final (una vez congelada).

## Advertencias para el informe

1. Matriz con 0.2 % de densidad y gran escasez de pares coevaluados; las métricas pueden ser muy bajas.
2. Las distribuciones de preferencias y destinos podrían ser artificiales; verificar en la documentación de origen si los registros son sintéticos, **sin inventar la conclusión**.
3. Los valores 1-3 se usan como no positivos explícitos, y los 4-5 como positivos por conveniencia experimental.
4. La aplicación es una demostración académica; la estimación de conversiones comerciales necesita supuestos externos, no se deriva automáticamente de `sigmoid`.
5. Se requiere completar figuras, discusión cuantitativa, cinco perfiles, concentración, comparación cold-start y matemáticas antes de declarar P4 terminado.

**Advertencia sobre test previo:** La primera etapa ya mostró métricas sobre `test`. Por tanto, no presentar ese conjunto como totalmente «intocado». Documentar esta consulta exploratoria y no utilizarla para cambiar arquitectura, umbral, semillas o hiperparámetros. Si el equipo exige una evaluación formal realmente sellada, deberá acordar un nuevo protocolo de partición antes de más experimentación y registrar el cambio, sin seleccionar el split que dé la métrica más alta.
