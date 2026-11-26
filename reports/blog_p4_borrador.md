# Módulo 3 — recomendación de destinos en India (borrador técnico)

> **Estado editorial:** borrador aportado por P4. Validar referencias APA 7, figuras, análisis de test final y si Kaggle clasifica los datos como reales o sintéticos antes de publicarlo.

## Problema y caso de uso

La aplicación ofrece destinos internacionales al usuario al consultar viajes: a un usuario existente se le muestran destinos no vistos, y a uno nuevo se le solicita una lista de preferencias. La recomendación devuelve un ranking Top-K, usualmente K=1–3 para el catálogo normalizado. El sistema es una demostración académica; no mide reservas ni conversiones comerciales observadas.

## Dataset y calidad

**Fuente exigida:** Aman Mehra, *Travel Recommendation Dataset*, Kaggle (enlace: https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset). Verificar en la página del autor la licencia, la procedencia y la etiqueta real/sintética; esta verificación continúa pendiente. **No concluir que son datos reales por contener nombres de usuarios**.

Los archivos descargados contenían 999 usuarios distintos, 1.000 IDs de destino y 1.998 pares usuario–ID únicos, con densidad 0,2 %. El catálogo tenía solo **cinco claves distintas** usando nombre, estado y tipo normalizados; cada una se repetía 200 veces con otros IDs. La política experimental agrupó esos IDs antes de separar interacciones. No se asume por ello identidad geográfica confirmada.

Tras agrupar, hay 1.644 pares usuario–lugar; 304 pares poseen varios eventos, de los cuales 160 muestran señales positivas y no positivas contradictorias y se excluyeron. La partición corregida (`seed=42`) contiene 373 positivos en train, 17 en validación, 90 en test y 81 en cold_test, pertenecientes a 66 usuarios nuevos. El método no es temporal global: las reseñas no tienen fecha, aunque el histórico de visitas sí. Todas las transformaciones, negativas observadas y frecuencias para entrenamiento se calculan sin leer etiquetas retenidas.

## Modelos y arquitectura

**Referencias:** (a) frecuencia de interacciones positivas en entrenamiento y (b) similitud coseno ítem–ítem sobre coocurrencias de entrenamiento, con respaldo de popularidad. **Complemento de cold-start:** emparejamiento de `Preferences` con `Type`, con popularidad de train solo como desempate.

**Red neuronal de dos torres:** produce un vector para usuario a partir de embedding de ID (índice 0 para nuevo) y vector multi-hot de preferencias, y otro para destino con embedding de ID y vector de categoría. Con redes densas independientes, obtiene dos representaciones de dimensión `q` y el logit de similitud escalado:

\[
\mathbf{h}_u=f_\theta([\mathbf{e}_u;\mathbf{p}_u]),\qquad
\mathbf{h}_d=g_\phi([\mathbf{e}_d;\mathbf{t}_d]),\qquad
z_{ud}=\frac{\mathbf{h}_u^T\mathbf{h}_d}{\sqrt{q}}.
\]

La puntuación usada para ranking es `sigmoid(z_ud)`, pero **no está calibrada como probabilidad de compra**. El modelo final corregido posee 15.041 parámetros entrenables; configuración seleccionada: embedding 16, capa oculta 32, latente 24, dropout 0,2, learning rate 0,005, AdamW con decaimiento 0,0001, lote 128, early stopping paciencia 4.

### Fundamentación probabilística de la pérdida

Se define la etiqueta `y=1` para calificaciones 4–5 y `y=0` para calificaciones explícitas 1–3. Si se asume `y | (u,d) ~ Bernoulli(p_ud)`, la verosimilitud de un ejemplo es `p_ud^y (1-p_ud)^(1-y)`. Su negativo logarítmico conduce a

\[
\mathcal L_{BCE}= -\frac1n\sum_{(u,d)}\big[y\log p_{ud}+(1-y)\log(1-p_{ud})\big].
\]

La implementación usa `BCEWithLogitsLoss` por estabilidad numérica. La relación 2:1 de negativos **submuestrea únicamente calificaciones explícitas 1–3**; nunca interpreta un par sin observar como rechazo verdadero. Los objetivos binarios son preferencia manifestada en valoraciones, no compra. El producto interno escalado y las capas de dos torres se optimizan mediante backpropagation y AdamW; explicar las ecuaciones de AdamW en la sección matemática común del blog.

### Métricas

Con un ranking \(R_u(K)\) y destinos relevantes \(G_u\):

\[
\mathrm{Precision@K}=\frac{|R_u(K)\cap G_u|}{K},\qquad
\mathrm{Recall@K}=\frac{|R_u(K)\cap G_u|}{|G_u|},
\]

\[
\mathrm{DCG@K}=\sum_{i=1}^K\frac{\mathrm{rel}_i}{\log_2(i+1)},\qquad
\mathrm{NDCG@K}=\frac{\mathrm{DCG@K}}{\mathrm{IDCG@K}},\qquad
\mathrm{Cobertura}=\frac{\#\text{destinos recomendados únicos}}{\#\text{destinos del catálogo}}.
\]

La concentración se expresa con HHI, suma de cuadrados de participaciones de exposición de los destinos recomendados. Con solo cinco destinos, el catálogo puede agotarse rápidamente; por tanto se reportan K=1,2,3 y el número de candidatos disponibles por usuario. No se comparan directamente métricas del experimento por ID (1.000 IDs) con las normalizadas (cinco entidades).

## Resultados de validación corregida (17 usuarios)

- `NDCG@1` popularidad = 0,17647; kNN = 0,17647; contenido = 0,05882.
- Búsqueda neuronal: 3 configuraciones, mejor validación `NDCG@1 = 0,70588` (trial 2, semilla 42).
- Configuración final neuronal repetida con 30 semillas: `NDCG@1` medio = **0,43725**, desviación estándar muestral = **0,14711**; mejor = 0,70588, peor = 0,17647. Cobertura media = 0,78 (desviación 0,15177). El modelo seleccionado por validación usa semilla 42; **no confundir su mejor desempeño con el promedio entre semillas**.
- Tiempo total de 30 entrenamientos = 29,289 segundos en el equipo de prueba del proyecto.

**Interpretación:** la red parece aportar capacidad de ranking en esta validación, pero solo 17 usuarios aportan positivos de validación, y la selección repetida sobre el mismo conjunto favorece resultados optimistas. No se ha probado validez comercial, robustez externa ni superioridad estadística con nueva muestra independiente. La naturaleza del dataset sigue pendiente de confirmar.

## Cold-start y negocio

El sistema compara para usuarios retenidos completamente en `cold_test` al menos dos estrategias sin historial: popularidad global de train y preferencias de contenido, además de la red con índice de usuario desconocido. Registrar sus resultados con K=1,2,3 mediante `report-p4 --split cold_test`, sin volver a ajustar hiperparámetros. Si `Recall@3` es muy alto, comprobar cuántas opciones elegibles tenía el usuario: no interpretarlo como rendimiento generalizable.

Un escenario exploratorio de campaña a Goa Beaches seleccionó 333 perfiles con preferencia de playas; suponiendo exposición 50 % y conversión 3 %, espera aproximadamente 5 conversiones. Es una **estimación de escenario**, no conversiones observadas ni salida calibrada de la red. La concentración de recomendaciones no se traduce directamente a demanda en rutas colombianas.

## Figuras y reproducibilidad

Generar las figuras con los datos y resultados **normalizados**, indicando como fuente «elaboración propia a partir de Kaggle» y las versiones de los comandos. La arquitectura, búsqueda de hiperparámetros, curvas NDCG de validación, BCE de entrenamiento y caja de 30 semillas son figuras diferentes. **No se generó pérdida BCE de validación**: para añadirla rigurosamente habría que reservar también etiquetas explícitas no positivas, ajustar la partición y reentrenar; no presentar NDCG como si fuera pérdida.

## Ética y limitaciones

No guardar ni publicar correo, nombres de usuarios ni texto libre de reseñas; el pipeline conserva solamente las columnas mínimas para modelamiento. La agregación de destinos puede confundir lugares distintos con igual nombre/estado/tipo: necesita validación geográfica si se amplía. Los datos no prueban preferencias reales de compradores ni conversiones; el catálogo reducido y la escasez de evidencia limitan la generalización. Documentar el uso de IA y al menos un error detectado con prompt literal, respuesta, evidencia y corrección, sin inventarlo.

## Pendientes de revisión antes de publicar

- Confirmar **con documentación del autor de Kaggle** si los registros son reales o sintéticos, licencia y fecha/versionado.
- Evaluar `cold_test` con los modelos congelados; después evaluar test warm una sola vez sin reajustar decisiones.
- Verificar que el material seleccionado por P5 en la web sea el experimento de entidades, no el antiguo por ID.
- Unificar referencias bibliográficas APA 7 verificadas y vincular cada figura en el texto del blog general.
- Resolver experimentalmente el requisito de pérdida de validación comparable con pérdida de entrenamiento (si el profesor lo exige de esa forma).
