# Diseño e Implementación de Filtros Digitales en Python

## 📌 Descripción del Proyecto

Este proyecto tiene como objetivo demostrar el diseño e implementación de filtros digitales utilizando Python. Se parte de una señal compuesta por varias frecuencias y ruido blanco, la cual se somete a distintos tipos de filtros (FIR e IIR) con el fin de analizar su comportamiento, respuesta y capacidad para eliminar componentes no deseadas.

Los filtros utilizados incluyen:
- Butterworth (IIR)
- Chebyshev Tipo I (IIR)
- Filtros FIR con diferentes configuraciones: pasa bajas, pasa altas y pasa banda

Además, se visualizan las señales antes y después del filtrado, así como la respuesta en frecuencia de cada filtro.

---

## 📂 Estructura del Código

El código está dividido en varias secciones principales:

1. **Generación de la señal de entrada**  
   Se crea una señal compuesta por 3 frecuencias (50 Hz, 150 Hz, 300 Hz) y ruido blanco.

2. **Diseño de los filtros**
   - Filtro pasa bajos Butterworth
   - Filtro pasa bajos FIR
   - Filtro pasa altos Chebyshev Tipo I
   - Filtro pasa altos FIR
   - Filtro pasa banda FIR

3. **Aplicación de los filtros**  
   Se aplica cada filtro a la señal original utilizando la función `lfilter`.

4. **Visualización de resultados**
   - Gráficas temporales de la señal original y filtradas
   - Respuestas en frecuencia de los filtros

---

## 🔧 Requisitos

Antes de ejecutar el código, asegúrate de tener instaladas las siguientes librerías de Python:

```bash
pip install numpy matplotlib scipy
