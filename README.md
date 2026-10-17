# Reconocedor de dígitos: CNN vs Red regular

Proyecto de la materia **Redes Neuronales**, Universidad de Sonora.

Dibuja un número del 0 al 9 y compara en vivo dos redes neuronales entrenadas con MNIST:

- **Red convolucional (CNN):** 163,850 parámetros
- **Red regular (densa):** 731,530 parámetros

Los modelos se entrenaron en Python (TensorFlow/Keras), se exportaron a TensorFlow.js y corren directamente en el navegador del celular.

- `index.html`: la página
- `modelos/`: los modelos exportados (`model.json` + pesos `.bin`)
- `entrenamiento/entrenar_y_exportar.py`: código de entrenamiento y exportación

Basado en el video de Ringa Tech *Usa tus modelos de Tensorflow en páginas web*.
