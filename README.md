# Convolución correlación y transformación
## Descripción
En el presente desarollo de laboratorio se tiene encuenta las siguientes operaciones y herramientas:
+ Convolución.
+ Correlación.
+ Transformada de Fourier.
  
Las cuales son fundamentales para operar entre señal y sistema, entre señales y para facilitar el analisis en el dominio de la freceuncia.
## Tener en cuenta
1. Se descarga una señal fisiologíca ECG de una sola derivación.
2. Se utilizan las siguientes librerias:
   + Wfdb.
   + Numpy.
   + Pandas.
   + Matplotlib.
3. Se utiliza **Jupyter NoteBook** para dividir el código en partes y trabajar en ellas sin importar el orden: escribir, probar funciones, cargar un archivo en la memoria y procesar el contenido. Con lenguaje de **Python**.
# Procedimiento
## Parte A.
Cada integrante debe crear un **sistema** apartir del código estudiantil y una **señal** con cada dígito de la Cédula de Ciudadania para aplicar la operación **CONVOLUCIÓN**; se realiza la gráfica secuencial, a mano y en lenguaje de **Python**.
#### *Integrante 1* 
<p align="center">
    <img src="https://github.com/user-attachments/assets/c61496ed-73e0-4d1e-b836-6da0b15439e5" 
         alt="Convolucion Ana" width="500">
    <br><em>Figura 1: Convolución 1 entre el sistema <strong>h(n)</strong> y la señal <strong>X[n]</strong> con su respectiva gráfica a mano.</em>
</p>


>Para programar en lenguaje de Python se utiliza la función de la libreria de Numpy que es **np.convolve([x(n),h(n)])** no importa el orden al poner los vectores dentro de la función porque la convolución es conmutativa. 
>
```python
xa=np.convolve([1, 0, 0, 0, 1, 8, 4, 1, 0, 5], [5, 6, 0, 0, 6, 5, 5])#Ana
print(xa)
>
y(n)= [ 5  6  0  0 11 51 73 29 12 78 99 66 25 35 25 25]
>

# Graficar la señal discreta 
plt.stem(range(len(xa)), xa, linefmt='g', markerfmt='go', basefmt='k')
plt.xlabel("muestras (n)")
plt.ylabel("Y(n)")
plt.title("Convolución Discreta, Ana")
plt.grid(True)
plt.show()
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/5f4a8ddf-b473-40a5-b8b8-ce5e727b19af" 
         alt="Convolucion ana grafica" width="500">
    <br><em>Figura 2: Gráfica Convolución Discreta **y(n)** vs **n**, integrante 1.</em>.</em>
</p>

#### *Integrante 2* 
<p align="center">
    <img src="https://github.com/user-attachments/assets/501d835b-cccb-4afb-a1f0-1b9da9383713" 
         alt="Convolucion lu" width="500">
    <br><em>Figura 3: Convolución 2 entre el sistema **h(n)** y la señal **X[n]** con su respectiva gráfica a mano .</em>
</p>

```python
xl=np.convolve([1, 0, 5, 4, 2, 8, 2, 8, 5, 8], [5, 6, 0, 0, 6, 6, 2])#Lunay
print(xl)
```
>y(n)=[  5   6  25  50  40  58  90 106 119 138 112  76  82  94  58  16]
>
```python
# Graficar la señal discreta 
plt.stem(range(len(xl)), xl, linefmt='r', markerfmt='ro', basefmt='k')
plt.xlabel("muestras (n)")
plt.ylabel("Y(n)")
plt.title("Convolución Discreta, Luna")
plt.grid(True)
plt.show()
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/41aa5325-09b7-4cbf-af48-dadaf824e404" 
         alt="Grafica convolucion Lu" width="500">
    <br><em>Figura 4: Gráfica Convolución Discreta **y(n)** vs **n**, integrante 2.</em>
</p>


#### *Integrante 3* 
<p align="center">
    <img src="https://github.com/user-attachments/assets/9fc8bc8b-41af-4e13-b1e3-336ce0d59925" 
         alt="Convolucion isa" width="500">
    <br><em>Figura 5: Convolución 3 entre el sistema **h(n)** y la señal **X[n]** con su respectiva gráfica a mano .</em>
</p>


```python
xi=np.convolve([1, 0, 1, 1, 0, 8, 3, 9, 5, 4], [5, 6, 0, 0, 6, 6, 7])#Isa
print(xi)
```
>y(n)= [  5   6   5  11  12  46  76  75  92 105  90 128 105 117  59  28]
>
```python
# Graficar la señal discreta 
plt.stem(range(len(xi)), xi)
plt.xlabel("muestras (n)")
plt.ylabel("Y(n)")
plt.title("Convolución Discreta, Isabela")
plt.grid(True)
plt.show()
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/6fddde56-58a3-47eb-8f07-ba0ac17f905a" 
         alt="conv  isa" width="500">
    <br><em>Figura 6: Gráfica Convolución Discreta **y(n)** vs **n**, integrante 3.</em>
</p>

## Parte B.
Se debe encontrar la correlación entre $X_1[nT_s] = \cos(2\pi 100 n T_s)$ y $X_2[nT_s] = \sin(2\pi 100 n T_s)$. Luego se debe encontrar la representación gráfica y secuencial.

Primero se almacena los valores en un DataFrame de la siguiente forma:
```python
fs=1/1.25e-3 # se define la frecuencia de muestreo
n=np.arange(0,10) # almacena valores entre 0 y 9
t= n/fs #conversión a tiempo real (t) donde se divide a n por la frecuencia de muestreo

x1=np.cos((2* np.pi*100*t))
x2=np.sin((2* np.pi*100*t))

df1=pd.DataFrame({
    'X1(n/fs)':x1,
    'X2(n/fs)':x2,
})
df1.head (10)
```
Se obtiene la siguiente tabla donde se tomaron valores de 0 a 9 para *n* debido al rango estipulado inicialmente para cada función.


|     | X1 (n/fs)| X2 (n/fs)|
|:-------:|:-----------------:|:----------:|
|    0    |      1.000000e+00  |   0.000000e+00   |        
|    1    | 7.071068e-01      |     7.071068e-01   |        
|      2  |     6.123234e-17	   |   1.000000e+00     |        
|     3   |      -7.071068e-01	   |     7.071068e-01   |        
|      4  |  -1.000000e+00      |      1.224647e-16  |         
|       5 |  -7.071068e-01       |   -7.071068e-01     |        
|        6|     -1.836970e-16	   |   -1.000000e+00     |        
|       7 |   7.071068e-01      |  -7.071068e-01      |        
|       8 |   1.000000e+00	     |   -2.449294e-16     |       
|       9|  7.071068e-01      |      7.071068e-01  |         


Apartir de esta tabla, se grafica las funciones de la siguiente forma: 

```python
fig, ax = plt.subplots(1, 1, figsize=(6,4)) # se crea la figura y los ejes
ax.scatter(x=df1['X1(n/fs)'], y=df1['X2(n/fs)'], alpha= 0.8) # genera un grafico de dispersion con los datos de df1
ax.set_xlabel('X1') 
ax.set_ylabel('X2');
```
Dando como resultado lo siguiente: 
<p align="center">
    <img src="grafica_x1_x2.png" 
         alt="grafica de x1 vs x2" width="500">
    <br><em>Figura 6: Gráfica de dispersión de **x1** vs **x2**.</em>
</p>

A continuación se calcula la correlación entre las señales **x1** y **x2** de la siguiente forma:
```python
#pearson
correlacion=df1.corr()
correlacion
```
Obteniendo como resultado:
|     | X1 (n/fs)| X2 (n/fs)|
|:-------:|:-----------------:|:----------:|
|    X1 (n/fs)  |     1.000000  |   0.078783  |        
|    X2 (n/fs) |0.078783     |     1.000000  |   


Para tener una mejor visualización sobre los resultados de la correlación, se hizo el siguiente diagrama:

  ```python
# Crear el heatmap
plt.figure(figsize=(6, 3))
sns.heatmap(correlacion, annot=True, cmap="coolwarm")

# Mostrar gráfico
plt.show()
```
<p align="center">
    <img src="correlacion.png" 
         alt="correlación" width="500">
    <br><em>Figura 7: Heatmap que permite observar la correlación entre **x1** y **x2**.</em>
</p>


Este diagrama nos permite visualizar con mayor facilidad la relación entre **x1 y x2**, por lo que se puede observar que **X1(n/fs)** con **X1(n/fs)** es igual a *1*  lo que quiere decir que cada variable tiene correlación perfecta consigo misma. Esto también pasa en la correlación **X2(n/fs)** con **X2(n/fs)**. En cuanto a **X1(n/fs)** con **X2(n/fs)** se obtiene un resultado de *0.079* lo que muestra que la correlación es muy mínima, casi nula, esto tiene sentido debido a que **cos(x)** y **sin(x)** estan desfasados y no siguen una relación lineal clara.

## Parte C.
En esta parte se analizará una señal electrocardiográfica (ECG) extraída del repositorio PhysioNet , específicamente de la base de datos ECG-ID Database . A continuación hablaremos de la caracterización y clasificación de la señal, así como un análisis estadístico detallado para evaluar sus propiedades tanto en el dominio del tiempo como en el dominio de la frecuencia, con modelos matemáticos como la transformada de Fourier y densidades espectrales tanto de potencia como de energía.

### Descripción de la señal:
+ De la base de datos se eligió el registro rec_15, que contiene 310 registros de ECG, obtenidos de 90 personas. Los datos corresponden a la derivación I del ECG, la señal se registra durante 20 segundos, digitalizada a 500 Hz con una resolución de 12 bits en un rango nominal de ±10 mV.
+ Los registros se obtuvieron de voluntarios (44 hombres y 46 mujeres de entre 13 y 75 años que eran estudiantes, colegas y amigos del autor).
+ El registro de La base de datos nos proporciona la señales sin procesar y  la filtrada de la siguiente manera:
+  Señal 0: ECG I (señal bruta)
+   Señal 1: ECG I filtrado (señal filtrada)
+   Una vez analizada la información procedemos a extraer los datos y graficar la señal y con ayuda de la libreria de pandas se hace un DataFrame en Jupyter notebook como compilador de Python.
Iniciamos con el llamado de las librerías usadas para graficar, extraer datos, realizar operaciones matemáticas, transformada de fourier y welch para el cálculo de la densidad espectral.

```python
## Librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
import numpy as np
from scipy.fft import fft  #  Importa correctamente la FFT
from scipy.signal import welch # Calcular la densidad espectral de potencia con Welch
```
Ahora, para exportar el documento y crear un data frame con los datos de la señal ECG filtrada (ECG I filtered) y el tiempo, creado a partir del número de muestras y la frecuencia.

```python
#Exportar documento
record = wfdb.rdrecord('rec_15')
print(record.__dict__)

# Convertir en DF global, o sea con todos los datos del documento
df_01 = pd.DataFrame(record.p_signal, columns=record.sig_name)
frecuencia = 500 #dada en el documento 
num_muestra= 10000.0 #dada en la data
tiempo= np.arange(0,num_muestra/frecuencia,1/frecuencia)
#SE CREA LA DATA FRAME SOLO PARA ECG I FILTERED Y TIEMPO
df_rt=df_01[['ECG I filtered']]

if len(tiempo) == len(df_rt):
    # Agregar la columna de tiempo al DataFrame
    df_rt["Tiempo (s)"] = tiempo
else:
    raise ValueError("El tamaño de    el array de tiempo no coincide con el número de filas del DataFrame.")
    
df_rt.head(60)
```
Ya obtenidas las 2 columnas con los datos de interés se crea una figura en donde se grafica la señal ECG filtrada como se muestra a continuación:

```python
#2.GRAFICAR SEÑAL
plt.figure(figsize=(17, 10))  # Configura el tamaño del gráfico

# Graficar la señal ECG fitrada en función del tiempo
plt.plot(df_rt["Tiempo (s)"], df_rt["ECG I filtered"], label="ECG", color="orange")

# Etiquetas y título
plt.title("ELECTROCARDIOGRAMA FILTRADO", fontsize=16)
plt.xlabel("Tiempo (s)", fontsize=14)
plt.ylabel("Amplitud (mV)", fontsize=14)

# Agregar una rejilla y la leyenda
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)

# Mostrar el gráfico
plt.show()
```
<p align="center">
    <img src="ecg.png" 
         alt="ECG" width="800">
    <br><em>Figura 8: Electrocardiograma.</em>
</p>

### 1.1 Cálculo estadísticos descriptivos:
+ Se calculó la media, la desviación estándar, la varianza, el coeficiente de variación y la frecuencia de muestreo, esta última se espera que coincida con la dada en el documento. En el código se desarrolla de manera automatizada (con funciones propias de Python) y con la lógica de programación, a continuación mostraremos el cálculo automatizado de los estadísticos:
  
  ```python
   #CALCULO DE # DE MUESTRAS - PROMEDIO - DESVIACIÓN ESTÁNDAR - VALOR MÍNIMO - CUARTILES - VALOR MÁXIMO
  df_01[['ECG I filtered']].describe().T
  #MEDIA
  np.mean(df_rt['ECG I filtered'])
  #DESVIACIÓN ESTÁNDAR
  np.std(df_rt['ECG I filtered'])
  #CÁLCULO PARA DESVIACIÓN ESTÁNDAR

  # Cálculo de las diferencias cuadradas
  diferencia = [(x - media) ** 2 for x in df_rt['ECG I filtered']]
  
  # Sumar las diferencias cuadradas
  sumatoria_diferencia = sum(diferencia)
  
  # Calcular la varianza (muestral)
  varianza = sumatoria_diferencia / (len(df_rt['ECG I filtered']) - 1)
  
  # Calcular la desviación estándar
  desviacion = varianza ** 0.5
  
  # Mostrar resultados
  print("Varianza: " + str(varianza))
  print("Desviación estándar: " + str(desviacion))
  
  # Calcular el intervalo de muestreo (suponiendo tiempo uniformemente espaciado)
  T_s = np.mean(np.diff(tiempo))  # Diferencia entre muestras consecutivas
  
  # Calcular la frecuencia de muestreo
  f_s = 1 / T_s
  
  print(f"Frecuencia de muestreo: {f_s:.2f} Hz")
  ```

+ ### Resultados
  
| Media de datos | Desviación Estandar  | Varianza | Coeficiente Variación| Frecuencia de muestreo|
|:-------:|:-----------------:|:----------:|:-----------------:|:-----------------:|
| 0.0012 |      0.1286     | 0.0165  |       10331.119   | 500

### 1.2 Caracterización de la señal:
+ La señal de ECG I filtered del registro de ECG-ID Database se puede clasificar según diferentes criterios:
+ Según su naturaleza, corresponde a una señal discreta, porque aunque el ECG es fisiológicamente contínua en el tiempo, en este caso ha sido muestreado digitalmente a 500 Hz, lo que lo convierte en una señal discreta en el dominio temporal.
+ Según el número de canales, es una señal unicanal, aunque el ECG contiene las 12 derivaciones, este registro solo contiene datos de la derivación I
+ Según el procesamiento de la seña, el registro contiene la señal sin filtrar como filtrada.
  
### 1.3 Aplicación FFT y gráficas densidad espectral potencia y energía:

+  La transformada rápida de Fourier (Fast Fourier Transform) convierte la señal del dominio del tiempo al dominio de la frecuencia, descomponiéndola en sus componentes sinusoidales [1] La PSD muestra cómo la potencia de la señal está distribuida en el espectro de frecuencias [2] La ESD muestra cómo la energía total de la señal se distribuye en frecuencia [3]

+  Se muestra el código implementado para calcular lo anterior:

### FFT
  ```python
# 1. Calcular la Transformada de Fourier (FFT)
# Obtener la señal ECG filtrada
ecg = df_rt["ECG I filtered"].values
Fs = 500 #frecuencia de muestreo dada en el documento
# Calcular la FFT
n = len(ecg)  # Número de puntos de la señal
frecuencias = np.fft.fftfreq(n, 1/Fs)  # Frecuencias correspondientes 1/Fs =Ts, vector de frecuencias
valfft = np.fft.fft(ecg)  # Valores de la FFT Devuelve un array de valores complejos que representan las componentes de frecuencia.

# Solo nos interesan las frecuencias positivasf
frecu_p = frecuencias[:n // 2] #frecp coeficientes de frecuencia positivos
fft_positivo = 2.0/n * np.abs(valfft[:n // 2]) #elimina la parte imaginaria y normaliza los valores de la FFT
```

Al calcular la FFT de la señal ECG, obtenemos un espectro simétrico con frecuencias positivas y negativas. Como solo nos interesan las positivas, tomamos la primera mitad de los valores. Sin embargo, esto reduce la energía total, por lo que multiplicamos por `2.0/n` para corregir la amplitud y reflejar correctamente la contribución de cada frecuencia en la señal original.

### Densidad espectral de energía (EDS)
  ```python
# Graficar la densidad espectral de energía (ESD)
plt.figure(figsize=(17, 10))
plt.plot(frecu_p, fft_positivo**2, label="Densidad Espectral de Energía (ESD)", color="blue")
plt.title("Densidad Espectral de Energía (ESD) de la señal ECG filtrada", fontsize=16)
plt.xlabel("Frecuencia (Hz)", fontsize=14)
plt.ylabel("Energía (mV²)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.show()
```
Se grafica la densidad, en el eje las frecuencias positivas, y en el eje x la transformada de Fourier normalizada elevada al cuadrado, que corresponde a la fórmula de la ESD, que indica cuánta energía hay en cada frecuencia.

<p align="center">
    <img src="densidad_espectral.png" 
         alt="Densidad espectral de energia" width="800">
    <br><em>Figura 9: Densidad Espectral de Energia.</em>
</p>

Como se observa en la figura 9 correspondiente a la densidad espectral de energía, observamos e inferimos que la mayor parte de la energía en la señal de ECG está en bajas frecuencias (menores a 50Hz), lo cual es lo esperado ya que la actividad eléctrica del corazón ocurre en este rango. Se observan picos menores a 10 Hz, posiblemente correspondientes a los ciclos del ECG (onda P, complejo QRS y onda T). A partir de 50 Hz, la energía disminuye rápidamente, lo que indica que se aplicó un filtro para eliminar el ruido de la red eléctrica, que en Colombia está en los 60 Hz. Como era esperado por el rango típico de frecuencias del ECG (0,05 - 100 Hz) no hay actividad significativa por encima de 100 Hz.

### Densidad espectral de potencia (PSD)
 ```python
# Calcular la densidad espectral de potencia (PSD) usando el método de Welch
frec_PSD, val_PSD = welch(ecg, fs=Fs, nperseg=1024) #nperseg=1024: Divide la señal en segmentos de 1024 muestras para promediar y reducir la varianza.

# 4. Graficar la Densidad Espectral de Potencia (PSD)
plt.figure(figsize=(17, 10))
plt.semilogy(frec_PSD, val_PSD, label="Densidad Espectral de Potencia (PSD)", color="green")
plt.title("Densidad Espectral de Potencia (PSD) de la señal ECG filtrada", fontsize=16)
plt.xlabel("Frecuencia (Hz)", fontsize=14)
plt.ylabel("Potencia/Frecuencia (mV²/Hz)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.show()
```
Utilizamos el método de Welch para calcular la PSD, el cual permite obtener una estimación más precisa y con menor ruido en comparación con el enfoque tradicional basado en la transformada de Fourier. Para ello, se emplea la función welch con la librería scipy.signal [4]. Este método consiste en dividir la señal en segmentos superpuestos, aplicar la transformada rápida de Fourier (FFT) a cada segmento y luego promediar los espectros de potencia obtenidos. Es decir, se calculan nuevas frecuencias para la PSD y así mismo el valor de cada potencia para cada frecuencia. En general, se divide la señal en segmentos de 1024 muestras (nperseg=1024), se aplica la transformada de Fourier de cada segmento y por último promedia los resultados para obtener la PSD.

Para graficar la señal se usa plt.semilogy para obtener en escala logarítmica en el eje 'y', esto es útil para visualizar la PSD, ya que los valores pueden variar en varios órdenes de magnitud. 

<p align="center">
    <img src="densidad_es_potencia.png" 
         alt="Densidad espectral de potencia" width="800">
    <br><em>Figura 10: Densidad Espectral de Potencia.</em>
</p>

La potencia de la señal ECG se concentra en bajas frecuencias y disminuye progresivamente. Se observa una pendiente descendente clara hasta 50 Hz, lo que indica que la mayor parte de la energía se encuentra en frecuencias fisiológicamente relevantes, asociadas con las distintas ondas del ECG, como la onda P, el complejo QRS y la onda T.  

A partir de 50 Hz, la densidad espectral de potencia es mucho menor, lo que sugiere que el filtrado ha reducido interferencias en altas frecuencias. Además, no se observan picos fuertes en el 60Hz lo que confirma que el ruido de la red eléctrica ha sido mitigado de manera efectiva. Más allá de 100 Hz, la energía es prácticamente despreciable, lo que indica que el ECG ha sido correctamente procesado y no hay información relevante en estas frecuencias.

### Estadísticos descriptivos frecuencia
Tal como calculamos los estadísticos descriptivos de la señal en el dominio del tiempo, los calcularemos con las frecuencias para posteriormente graficarlo.

 ```python
# 1. Obtener las frecuencias y los valores de la FFT
frecu_p = frequencies[:n // 2]  # Frecuencias positivas
fft_positivo = 2.0/n * np.abs(fft_values[:n // 2])  # Magnitudes de la FFT

# 2. Calcular los estadísticos

# Frecuencia media
media_frec = np.sum(frecu_p * fft_positivo) / np.sum(fft_positivo)

# Frecuencia mediana
suma = np.cumsum(fft_positivo)  # Suma acumulada de las magnitudes de la FFT
xm = np.where(cumulative_sum >= cumulative_sum[-1] / 2)[0][0] # Índice donde se alcanza la mitad de la energía
mediana_frec = frecu_p[xm]  # Frecuencia correspondiente al índice

# Desviación estándar de la frecuencia
desviacion_frec = np.sqrt(np.sum(fft_positivo * (frecu_p - media_frec)**2) / np.sum(fft_positivo))

print(f"Frecuencia media: {media_frec:.2f} Hz")
print(f"Frecuencia mediana: {mediana_frec:.2f} Hz")
print(f"Desviación estándar de la frecuencia: {desviacion_frec:.2f} Hz")
 ```
El histograma de frecuencias obtenido mediante la FFT de la señal ECG muestra que la mayor parte de la energía está concentrada en bajas frecuencias, con una disminución progresiva a medida que la frecuencia aumenta, lo cual es característico de señales biológicas. A partir de 50 Hz, la magnitud se reduce notablemente, indicando que se aplicó un filtrado efectivo para eliminar interferencias de alta frecuencia. Además, no se observan picos en 60Hz, lo que confirma la eliminación del ruido de la red eléctrica y sugiere que la señal ha sido correctamente  procesada.

<p align="center">
    <img src="histograma_f.png" alt="Histograma de Frecuencia" width="800">
    <br><em>Figura 11: Histograma de Frecuencia</em>
</p>

## Referencias

[1] Transformada rápida de Fourier FFT | Academia Svantek. (2023, December 20). Svantek.
[2] Qué es: análisis espectral. (2024, July 25). LEARN STATISTICS EASILY. https://es.statisticseasily.com/glosario/%C2%BFQu%C3%A9-es-el-an%C3%A1lisis-espectral%3F/
[3] (N.d.). Edu.Ar. Retrieved February 21, 2025, from https://www.fceia.unr.edu.ar/tesys/html/densidad_espectral_energia_bw.pdf









