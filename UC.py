import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, firwin, lfilter, freqz

# ----------------------------
# Parámetros generales
# ----------------------------

fs = 500  # Frecuencia de muestreo (Hz)
t = np.linspace(0, 1.0, fs, endpoint=False)  # Vector de tiempo de 1 segundo

# ----------------------------
# Señal original + Ruido
# ----------------------------

# Señal compuesta de 3 senoidales (frecuencias: 5Hz, 50Hz y 120Hz)
frecuencia1 = 5
frecuencia2 = 50
frecuencia3 = 120
senal = np.sin(2*np.pi*frecuencia1*t) + np.sin(2*np.pi*frecuencia2*t) + np.sin(2*np.pi*frecuencia3*t)

# Agregar ruido blanco
ruido = np.random.normal(0, 0.5, senal.shape)
senal_ruido = senal + ruido

# ----------------------------
# Función para aplicar filtro y graficar
# ----------------------------

def aplicar_filtro(b, a, signal, tipo):
    """
    Aplica un filtro (definido por b, a) a la señal y grafica la respuesta en frecuencia
    y la señal filtrada.
    """
    # Respuesta en frecuencia del filtro
    w, h = freqz(b, a, worN=8000)
    plt.figure(figsize=(12, 4))
    plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b')
    plt.title(f'Respuesta en Frecuencia - {tipo}')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Ganancia')
    plt.grid()
    plt.show()

    # Aplicar filtro
    senal_filtrada = lfilter(b, a, signal)

    # Graficar señal filtrada
    plt.figure(figsize=(12, 4))
    plt.plot(t, signal, 'gray', alpha=0.5, label='Señal con ruido')
    plt.plot(t, senal_filtrada, 'r', label='Señal filtrada')
    plt.title(f'Señal filtrada - {tipo}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid()
    plt.show()

# ----------------------------
# 1. Filtro Pasa bajos - Butterworth
# ----------------------------

# Diseño del filtro
orden = 6
frecuencia_corte = 30  # Hz
b, a = butter(orden, frecuencia_corte/(fs/2), btype='low')
aplicar_filtro(b, a, senal_ruido, 'Pasa bajos Butterworth')

# ----------------------------
# 2. Filtro Pasa altos - Chebyshev Tipo I
# ----------------------------

orden = 6
frecuencia_corte = 60  # Hz
b, a = cheby1(orden, 1, frecuencia_corte/(fs/2), btype='high')
aplicar_filtro(b, a, senal_ruido, 'Pasa altos Chebyshev I')

# ----------------------------
# 3. Filtro Pasa bandas - FIR con ventana
# ----------------------------

frecuencia_banda = [40, 100]
num_taps = 101
b = firwin(num_taps, [frecuencia_banda[0]/(fs/2), frecuencia_banda[1]/(fs/2)], pass_zero=False)
a = 1  # FIR: solo coeficiente b
aplicar_filtro(b, a, senal_ruido, 'Pasa bandas FIR (ventana)')

# ============================================================
# SEGUNDA PARTE: Visualización comparativa de filtros
# ============================================================

# Funciones para diseñar filtros Butterworth
def butter_lowpass(cutoff, fs, order=6):
    return butter(order, cutoff / (fs / 2), btype='low')

def butter_highpass(cutoff, fs, order=6):
    return butter(order, cutoff / (fs / 2), btype='high')

def butter_bandpass(lowcut, highcut, fs, order=6):
    return butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')

# Aplicar filtros
b_lp, a_lp = butter_lowpass(30, fs)
senal_filtrada_lp = lfilter(b_lp, a_lp, senal_ruido)

b_hp, a_hp = butter_highpass(60, fs)
senal_filtrada_hp = lfilter(b_hp, a_hp, senal_ruido)

b_bp, a_bp = butter_bandpass(40, 100, fs)
senal_filtrada_bp = lfilter(b_bp, a_bp, senal_ruido)

# ----------------------------
# Gráfica 1: Señal original vs señal con ruido
# ----------------------------
plt.figure(figsize=(12, 4))
plt.plot(t, senal, label='Señal original')
plt.plot(t, senal_ruido, alpha=0.5, label='Señal con ruido')
plt.title('Señal Original vs Señal con Ruido')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# ----------------------------
# Gráfica 2: Filtrado pasa bajos
# ----------------------------
plt.figure(figsize=(12, 4))
plt.plot(t, senal_ruido, alpha=0.5, label='Señal con ruido')
plt.plot(t, senal_filtrada_lp, 'g', label='Filtrada (Pasa bajos)')
plt.title('Filtrado Pasa Bajos')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# ----------------------------
# Gráfica 3: Filtrado pasa altos
# ----------------------------
plt.figure(figsize=(12, 4))
plt.plot(t, senal_ruido, alpha=0.5, label='Señal con ruido')
plt.plot(t, senal_filtrada_hp, 'm', label='Filtrada (Pasa altos)')
plt.title('Filtrado Pasa Altos')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()

# ----------------------------
# Gráfica 4: Filtrado pasa bandas
# ----------------------------
plt.figure(figsize=(12, 4))
plt.plot(t, senal_ruido, alpha=0.5, label='Señal con ruido')
plt.plot(t, senal_filtrada_bp, 'c', label='Filtrada (Pasa bandas)')
plt.title('Filtrado Pasa Bandas')
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.legend()
plt.grid()
plt.show()
