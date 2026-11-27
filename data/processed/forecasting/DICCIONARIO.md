# Diccionario de datos M1 (pronóstico de demanda)

Generado por `python -m transport_ai.forecasting.cli --mode data` a partir de `data/raw/operacion_pasajeros.csv`
(SHA-256 y conteos en `auditoria_crudo.json` y `data/manifests/forecast_manifest.json`).

## Fuente
- **Operación de pasajeros y despacho de vehículos en la modalidad de transporte de pasajeros por carretera**,
  Superintendencia de Transporte, portal datos.gov.co. Archivo local: 32 669 785 filas, columnas
  `TERMINAL, CLASE_VEHICULO, NIVEL_SERVICIO, MUNICIPIO_ORIGEN_RUTA, MUNICIPIO_DESTINO_RUTA, FECHA_DESPACHO,
  HORA_DESPACHO, TIPO_DESPACHO, DESPACHOS, PASAJEROS`. Cada fila = terminal × clase × nivel × ruta × fecha × hora × tipo.
  Recurso `eh75-8ah6` (https://www.datos.gov.co/d/eh75-8ah6), publicado por Superintendencia de Transporte - SUPERTRANSPORTE,
  licencia **Creative Commons Attribution-ShareAlike 4.0**; última actualización de datos según el catálogo: 2026-05-03
  (consultado 2026-11-27). La copia local se identifica por su SHA-256, no por la fecha del portal.
- **DIVIPOLA - Códigos municipios** (DANE), recurso `gdxc-w37w` (https://www.datos.gov.co/d/gdxc-w37w), licencia CC BY-SA 4.0,
  descargado el 2026-11-27 a `data/raw/divipola/divipola.csv`.
  Los códigos se leen como texto (5 dígitos, ceros iniciales conservados). Las 1 856 rutas del periodo tienen ambos códigos en DIVIPOLA.

## Archivos
| Archivo | Contenido |
|---|---|
| `diario_agregado.parquet` | Suma diaria por `fecha, origen, destino, tipo` de pasajeros, despachos, registros, `reg_pax0` (registros con 0 pasajeros y despachos > 0), `reg_exceso`/`pax_exceso` (registros con > 80 pasajeros por despacho). Todo el periodo, todas las rutas. |
| `terminales_dia.parquet` | Registros por terminal y día (detección de días sin reporte nacional). |
| `auditoria_crudo.json` | Filas, inválidas, duplicados exactos, registros por año, hash del crudo. |
| `anomalias_crudo.csv` | Filas descartadas por fecha imposible (años 2222 y 6202) o valores no numéricos. |
| `series.parquet` | **Serie modelada**: 20 rutas seleccionadas × 1 095 días (2021-01-01..2023-12-31). |
| `rutas.parquet` | Cobertura de las 1 856 rutas dirigidas, criterio de aptitud, selección y nombres DIVIPOLA. |
| `anomalias_series.csv` | Días desconocidos y atípicos de las rutas seleccionadas, con su regla. |

## `series.parquet`
| Columna | Tipo | Significado |
|---|---|---|
| `route_id` | texto | `ORIGEN-DESTINO` con códigos DIVIPOLA; dirigida (el regreso es otra serie) |
| `fecha` | fecha | día calendario (calendario continuo, sin huecos) |
| `pasajeros` | float | **objetivo**: pasajeros/día con `TIPO_DESPACHO = ORIGEN`; **NaN = desconocido** |
| `pasajeros_sin_limpiar` | float | suma sin tratamiento (para sensibilidad) |
| `despachos`, `registros` | float | despachos y filas crudas del día (NaN si no hubo registros) |
| `reg_pax0`, `reg_exceso` | float | registros con 0 pasajeros y despacho > 0; registros sobre capacidad |
| `anomalia` | texto | `""`, `sin_registro` o `cero_con_despacho` |
| `valido` | bool | el día entra a entrenamiento/evaluación |
| `atipico` | bool | fuera de [0.2, 5] × mediana móvil de 28 días; se marca, **no** se modifica |

## Reglas de limpieza (y por qué)
1. **Día sin registros ≠ cero.** Se deja NaN; la pérdida y las métricas lo ignoran y las redes reciben una máscara.
2. **Cero pasajeros con despachos > 0 en el día** → NaN: hubo operación, pero no se registraron pasajeros.
3. **Duplicados exactos (6,4 M filas, 19,6 %) se conservan.** La tasa es estable entre meses (17-23 %) y cae con el número
   de pasajeros (37 % con 0 pax, 4 % con 20-80 pax). Una doble carga daría una tasa uniforme. Lo coherente es que sean
   despachos distintos que comparten la clave hora × clase × terminal (el archivo no trae empresa ni vehículo).
4. **Exceso de capacidad** (> 80 pasajeros por despacho): 0 registros en el archivo; la regla queda activa.
5. **Fechas imposibles** (3 filas) se descartan; 2019-2020 y 2025-2026 tienen < 0,1 % de las filas y 2024 no existe:
   el periodo de estudio es 2021-2023, con 37-51 terminales reportando todos los días (0 días sin reporte nacional).
6. **Selección de rutas:** cobertura válida ≥ 98 %, hueco máximo ≤ 3 días, media ≥ 100 pax/día (175 rutas aptas de 1 856);
   se modelan las 20 de mayor volumen con un modelo global. El número 20 (no 3-5) da a las redes suficientes ventanas
   (~10 400 de entrenamiento) y permite reportar error por ruta; P2 puede usar cualquier subconjunto.
