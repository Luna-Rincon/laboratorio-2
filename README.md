# Convolución correlación y transformación
## Descripción
En el presente desarollo de laboratorio se tiene encuenta las siguientes operaciones y herramientas:
+ Convolución.
+ Correlación.
+ Transformada de Fourie.
  
Las cuales son fundamentales para operar entre señal y sistema, entre señales y para facilitar el analisis en el dominio de la freceuncia.
## Tener en cuenta
1. Se descarga una señal fisiologíca EEG de una sola derivación.
2. Se utilizan las siguientes librerias:
   + Wfdb.
   + Numpy.
   + Pandas.
   + Matplotlib.
3. Se utiliza **Jupyter NoteBook** para dividir el código en partes y trabajar en ellas sin importar el orden: escribir, probar funciones, cargar un archivo en la memoria y procesar el contenido. Con lenguaje de **Python**.
## Procedimiento
### Parte A.
Cada integrante debe crear un **sistema** apartir del código estudiantil y una **señal** con cada dígito de la Cédula de Ciudadania para aplicar la operación **CONVOLUCIÓN**; se realiza la gráfica secuencial, a mano y en lenguaje de **Python**.
#### *Integrante 1* 
![Convolucion Ana ](https://github.com/user-attachments/assets/c61496ed-73e0-4d1e-b836-6da0b15439e5)
<br><em>Figura 1: Convolución 1 entre el sistema **h(n)** y la señal **X[n]** con su respectiva gráfica a mano .</em></p>

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
![Convolucion ana grafica](https://github.com/user-attachments/assets/5f4a8ddf-b473-40a5-b8b8-ce5e727b19af)
<br><em>Figura 2: Gráfica Convolución Discreta **y(n)** vs **n**, integrante 1.</em></p>

#### *Integrante 2* 

![Convolucion lu](https://github.com/user-attachments/assets/501d835b-cccb-4afb-a1f0-1b9da9383713)
<br><em>Figura 3: Convolución 2 entre el sistema **h(n)** y la señal **X[n]** con su respectiva gráfica a mano .</em></p>
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
![Grafica convolucion Lu](https://github.com/user-attachments/assets/41aa5325-09b7-4cbf-af48-dadaf824e404)
<br><em>Figura 4: Gráfica Convolución Discreta **y(n)** vs **n**, integrante 2.</em></p>

#### *Integrante 3* 
![Convolu isa](https://github.com/user-attachments/assets/9fc8bc8b-41af-4e13-b1e3-336ce0d59925)
<br><em>Figura 5: Convolución 3 entre el sistema **h(n)** y la señal **X[n]** con su respectiva gráfica a mano .</em></p>

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
![conv  isa](https://github.com/user-attachments/assets/6fddde56-58a3-47eb-8f07-ba0ac17f905a)
<br><em>Figura 6: Gráfica Convolución Discreta **y(n)** vs **n**, integrante 3.</em></p>
