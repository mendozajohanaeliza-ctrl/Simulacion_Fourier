import numpy as np
import matplotlib.pyplot as plt


# 1. CONFIGURACIÓN INICIAL DE PARÁMETROS

Fs = 1000.0           # Frecuencia de muestreo (Hz)
T = 1.0 / Fs          # Periodo de muestreo
N = 2000              # Número de puntos (2 segundos de señal)
t = np.arange(-N//2, N//2) * T  # Vector de tiempo centrado en cero


# 2. DEFINICIÓN DE SEÑALES ELEMENTALES

# Señal A: Pulso Rectangular (Ancho de 0.2 segundos)
ancho_pulso = 0.2
pulso_rect = np.where(np.abs(t) <= ancho_pulso / 2, 1.0, 0.0)

# Señal B: Función Senoidal (Frecuencia de 10 Hz)
f0 = 10.0
senal_seno = np.sin(2 * np.pi * f0 * t)


# 3. CÁLCULO DE LA TRANSFORMADA DE FOURIER (FFT)

# Usamos fftshift para centrar la frecuencia 0 Hz en el medio del gráfico
frecuencias = np.fft.fftshift(np.fft.fftfreq(N, T))

# FFT del Pulso Rectangular
X_rect = np.fft.fftshift(np.fft.fft(pulso_rect)) * T  # Escalado por T para aproximar CTFT
magnitud_rect = np.abs(X_rect)
fase_rect = np.angle(X_rect)

# FFT de la Senoidal
X_seno = np.fft.fftshift(np.fft.fft(senal_seno)) * T
magnitud_seno = np.abs(X_seno)
fase_seno = np.angle(X_seno)


# 4. COMPROBACIÓN DE PROPIEDADES (Desplazamiento)

# Desplazamos el pulso rectangular en el tiempo (t0 = 0.3 segundos)
t0 = 0.3
pulso_desplazado = np.where(np.abs(t - t0) <= ancho_pulso / 2, 1.0, 0.0)
X_desplazado = np.fft.fftshift(np.fft.fft(pulso_desplazado)) * T


# 5. VISUALIZACIÓN DE RESULTADOS


# Gráfico 1: Pulso Rectangular y su Espectro (Sinc)
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(t, pulso_rect, 'b', lw=2)
plt.title("Pulso Rectangular (Tiempo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(frecuencias, magnitud_rect, 'r', lw=2)
plt.title("Espectro de Magnitud (Función Sinc)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("|X(f)|")
plt.xlim(-30, 30)
plt.grid()

# Gráfico 2: Senoidal y su Espectro (Impulsos)
plt.subplot(2, 2, 3)
plt.plot(t, senal_seno, 'g')
plt.title("Señal Senoidal 10 Hz (Tiempo)")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.xlim(-0.2, 0.2) # Zoom para ver los ciclos
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(frecuencias, magnitud_seno, 'm', lw=2)
plt.title("Espectro de Magnitud (Impulsos en $\pm 10$ Hz)")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("|X(f)|")
plt.xlim(-20, 20)
plt.grid()

plt.tight_layout()
plt.show()

# Gráfico 3: Demostración de Propiedad de Desplazamiento
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(frecuencias, np.abs(X_rect), 'r', label='Original', lw=2)
plt.plot(frecuencias, np.abs(X_desplazado), 'k--', label='Desplazado', lw=1.5)
plt.title("Magnitud del Espectro (No cambia)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(-20, 20)
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(frecuencias, np.angle(X_rect), 'r', label='Fase Original')
plt.plot(frecuencias, np.angle(X_desplazado), 'k', label='Fase Desplazada', alpha=0.7)
plt.title("Fase del Espectro (Cambio lineal)")
plt.xlabel("Frecuencia (Hz)")
plt.xlim(-20, 20)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
