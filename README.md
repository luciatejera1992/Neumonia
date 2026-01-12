# Detección de neumonía por visión por computador

Proyecto de ejemplo que aplica redes neuronales convolucionales para detectar neumonía en imágenes de rayos X torácicos. Incluye el notebook de entrenamiento, modelos guardados y gráficos con métricas y resultados.

## Contenido

Este repositorio contiene:

- `neumonia.ipynb` — Notebook principal con el pipeline: carga de datos, preprocesado, aumentación, definición del modelo, entrenamiento y evaluación.
- `pneumoniamnist.npz` — Conjunto de datos (preprocesado) usado en el notebook.
- `mejor_modelo_pneumonia.keras` — Modelo final entrenado (formato Keras SavedModel / HDF5 según la exportación).
- `modelos_guardados/modelo_pneumonia.keras` — Copia adicional del modelo guardado.
- `RedesNeuronales:Neumonia/` — Carpeta con imágenes generadas (gráficas, ejemplos y resultados):
	- `historial_entrenamiento.png` — Curvas de pérdida/accuracidad durante el entrenamiento.
	- `matriz_confusion.png` — Matriz de confusión final.
	- `ejemplos_imagenes.png`, `ejemplos_predicciones.png` — Ejemplos visuales de la entrada y predicciones.
	- `distribucion_clases.png`, `distribucion_probabilidades.png` — Gráficas exploratorias de los datos.

## Descripción breve

El objetivo es entrenar una red neuronal convolucional que clasifique radiografías en dos clases: `NORMAL` y `PNEUMONIA`. El repositorio incluye:

- Un notebook reproducible con todo el flujo (preprocesado, augmentación, definición del modelo y evaluación).
- Modelos guardados listos para inferencia.
- Visualizaciones que permiten revisar el desempeño y la distribución de los datos.

## Requisitos

Recomendado: Python 3.8+ y GPU opcional (mejora el tiempo de entrenamiento). Dependencias principales:

- tensorflow
- numpy
- matplotlib
- scikit-learn
- pillow
- jupyter
- seaborn

Puedes instalarlas en un entorno virtual con:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tensorflow numpy matplotlib scikit-learn pillow jupyter seaborn
```

Si vas a usar GPU, instala la versión de `tensorflow` compatible con tu CUDA/cuDNN.

## Cómo reproducir el entrenamiento

1. Activar el entorno (ver arriba).
2. Abrir `neumonia.ipynb` en Jupyter Notebook / Jupyter Lab:

```bash
jupyter notebook pneumonia.ipynb
```

3. Ejecutar las celdas en orden. El notebook está organizado para:
	 - Cargar `pneumoniamnist.npz`.
	 - Dividir en train/val/test.
	 - Aplicar aumentación de datos (si está activada).
	 - Definir y compilar el modelo (arquitectura CNN estándar + capas densas finales).
	 - Entrenar y guardar el mejor modelo en `mejor_modelo_pneumonia.keras`.

Nota: el entrenamiento puede tardar dependiendo del hardware. Para resultados idénticos usa semillas fijas (ver notebook) y la misma versión de librerías.

## Uso / Inferencia (ejemplo)

Ejemplo mínimo para cargar el modelo y predecir una imagen con TensorFlow/Keras:

```python
from tensorflow import keras
from PIL import Image
import numpy as np

# Cargar modelo
model = keras.models.load_model('mejor_modelo_pneumonia.keras')

# Cargar y preprocesar imagen (ajustar tamaño según el modelo).
img = Image.open('ruta/a/una/rx.jpg').convert('L')  # convertir a escala de grises si corresponde
img = img.resize((224, 224))  # ejemplo: 224x224 — adaptar si el modelo espera otro tamaño
arr = np.asarray(img) / 255.0
arr = arr.reshape(1, arr.shape[0], arr.shape[1], 1)  # añadir batch y canal

# Predecir
pred = model.predict(arr)
prob = pred[0][0]
print('Probabilidad de neumonía:', prob)
```

Adapta el preprocesado (canales, normalización y tamaño) al que uses en el notebook.

## Resultados y métricas

Revisa las imágenes en `RedesNeuronales:Neumonia/` para ver las curvas de entrenamiento, la matriz de confusión y ejemplos. En el notebook se calculan métricas clave como precisión, recall, F1 y AUC.

## Estructura del proyecto

Raíz del repo (resumen):

```
README.md
neumonia.ipynb
pneumoniamnist.npz
mejor_modelo_pneumonia.keras
modelos_guardados/
RedesNeuronales:Neumonia/
```
