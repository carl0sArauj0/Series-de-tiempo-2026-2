
---

### Esquema General de Intercalación

```text
[Inicio]
   │
   ├── PARTE 1 (Persona A): El Reto de la Tendencia (6 Técnicas) y Estacionalidad
   │                                  ↓ (Pase de palabra)
   ├── PARTE 2 (Persona B): Estrategias Multi-Paso, Pipeline y MODELO 1: Árbol de Decisión
   │                                  ↓ (Pase de palabra)
   ├── PARTE 3 (Persona A): MODELO 2: Perceptrón Multicapa (MLP) y su Búsqueda
   │                                  ↓ (Pase de palabra)
   └── PARTE 4 (Persona B): MODELO 3: Red LSTM, Comparación Final, Producción y Conclusiones
   │
[Cierre]
```

---

### PARTE 1: Persona A — El Reto de la Tendencia (6 Técnicas) y Estacionalidad
> **Enfoque:** Diagnóstico cuantitativo de la serie y justificación metodológica de por qué se modela la serie diferenciada frente a otros métodos de tendencia.

* **1. Comparación de 6 técnicas de estimación de tendencia:**
  * Mostrar y contrastar la gráfica multitécnica:
    1. *Regresión Lineal OLS.*
    2. *Regresión Polinomial Cuadrática.*
    3. *Filtro Hodrick-Prescott ($\lambda = 14\,400$).*
    4. *LOESS (regresión local).*
    5. *Splines Cúbicos de Suavizamiento (`UnivariateSpline`).*
    6. *Descomposición STL.*
* **2. Por qué gana la diferenciación ($\Delta Y_t$):**
  * Mostrar la tabla de propiedades de los residuos.
  * Argumento clave: Los métodos de suavizado (Splines, HP, LOESS) sufren de **efectos de borde** y exigen un modelo secundario para extrapolar al futuro (riesgo de *data leakage*). La primera diferencia remueve la raíz unitaria sin estimar parámetros.
* **3. Dependencia y estructura periódica de $\Delta Y_t$:**
  * Análisis de ACF, PACF y correlación de distancia (dCor): dependencia fuerte en rezagos cortos ($1, 2, 3$) y estacional ($12$).
  * Periodograma: ciclos marcados en 12, 6, 4 y 3 meses (armónicos del ciclo anual que definirán las entradas).

> **Frase de transición a Persona B:**  
> *"Una vez fundamentada la decisión de trabajar con la serie diferenciada $\Delta Y$, el reto es cómo proyectar a 6 meses sin perder la inercia acumulada. Mi compañero/a explicará el diseño matemático del pronóstico y nuestro primer modelo: el Árbol de Decisión."*

---

### PARTE 2: Persona B — Estrategias Multi-Paso, Pipeline y MODELO 1: Árbol de Decisión
> **Enfoque:** La matemática de volver a niveles, la arquitectura sin fuga (*gap* de 5 meses) y la evaluación completa del árbol.

* **1. La suma telescópica vs. la trampa de $\Delta Y_{t+6}$:**
  * Explicar por qué no se puede pronosticar solo $\Delta Y_{t+6}$ (cambio de un solo mes que ignora los intermedios e incurre en fuga si se suma a datos futuros).
  * La fórmula correcta: $Y_{t+6} = Y_t + \sum_{j=1}^6 Z_{t+j}$.
* **2. Las 4 estrategias multi-paso y el diseño del "Gap":**
  * Definición de: *Recursiva*, *Directa a $h=6$*, *Directa con 6 modelos* y *MIMO (One-Shot)*. Benchmark: Persistencia.
  * La regla del hueco (*gap* de 5 meses): asegurar que la respuesta a 6 meses del entrenamiento no invada el bloque de validación.
  * Validación Simple (afectada por el choque de 2020) vs. Secuencial (ventanas crecientes).
* **3. MODELO 1: Árbol de Decisión (`DecisionTreeRegressor`):**
  * Entradas: Matriz tabular con rezagos $[1, 2, 3, 4, 13]$ y 11 *dummies* mensuales.
  * Curva de validación para `max_depth` y búsqueda de hiperparámetros (*Grid* y *Random*).
  * Resultados del árbol: ¿En qué estrategia funciona mejor? (ventaja de la recursiva en árboles).

> **Frase de transición a Persona A:**  
> *"El árbol nos da una línea base sólida y no paramétrica. Sin embargo, para capturar interacciones no lineales continuas pasamos a las redes neuronales. Mi compañero/a presentará el segundo modelo: el Perceptrón Multicapa."*

---

### PARTE 3: Persona A — MODELO 2: Perceptrón Multicapa (MLP) y su Búsqueda
> **Enfoque:** La transición a Deep Learning tabular, preprocesamiento riguroso, parada temprana y comportamiento del MLP.

* **1. Qué cambia en el MLP frente al Árbol:**
  * **Estandarización estricta:** Cálculo de $\mu$ y $\sigma$ únicamente con el bloque inicial de entrenamiento (antes de 2015) para no contaminar ningún pliegue; las *dummies* se mantienen binarias ($0/1$).
  * Salida lineal y optimizador Adam con pérdida MSE.
* **2. Control del entrenamiento y parada temprana:**
  * Uso de *EarlyStopping* con paciencia para seleccionar automáticamente el número óptimo de épocas y evitar el sobreajuste.
  * Curvas de aprendizaje del MLP elemental: cómo evoluciona la pérdida en entrenamiento vs. validación.
* **3. Búsqueda de hiperparámetros del MLP:**
  * Comparación de métodos de búsqueda: *Grid*, *Random* y *Hyperband* (evaluación por rondas donde sobrevive el mejor tercio).
  * Espacio explorado: número de capas, neuronas, activación (`relu` vs `tanh`), tasa de aprendizaje y tamaño de lote.
* **4. Desempeño del MLP:**
  * Resultados por estrategia: ¿Es mejor la versión MIMO (6 salidas simultáneas) o la directa acumulada para una red densa?
  * Sensibilidad del MLP ante el bloque de validación simple vs. secuencial.

> **Frase de transición a Persona B:**  
> *"El MLP trata la ventana como un vector estático de variables independientes. Para explotar formalmente el orden temporal de los datos, pasamos a las redes recurrentes. Mi compañero/a expondrá la LSTM, la comparación final entre los tres modelos, la producción y las conclusiones."*

---

### PARTE 4: Persona B — MODELO 3: Red LSTM, Comparación Final, Producción y Conclusiones
> **Enfoque:** Arquitectura recurrente, contraste entre los tres modelos, pronóstico real fuera de muestra y balance final.

* **1. MODELO 3: Red Recurrente LSTM:**
  * **Transformación 3D de la entrada:** Paso de tabla plana a un tensor $(N, 13, 3)$ [muestras, pasos temporales, variables].
  * **Codificación continua del tiempo:** En cada paso entran el valor estandarizado y las funciones trigonométricas $(\sin, \cos)$ del mes para proyectar la estacionalidad en un círculo continuo.
  * Dinámica de la memoria recurrente, sintonización con Hyperband y resultados propios de la LSTM.
* **2. Comparación Final Honesta (Árbol vs. MLP vs. LSTM):**
  * Mostrar la tabla de resultados consolidados evaluados en el conjunto de prueba (elegidos estrictamente con validación secuencial).
  * Desglose del error por horizonte ($h=1, \dots, 6$): ¿Qué modelo sostiene mejor la trayectoria y cómo se comparan frente a la persistencia?
* **3. Pronóstico en Producción (Julio a Diciembre de 2026):**
  * Reentrenamiento de los modelos ganadores con el 100% de la historia observada (hasta junio de 2026).
  * Mostrar el gráfico de producción con la trayectoria proyectada para el segundo semestre de 2026.
* **4. Conclusiones y Lecciones Metodológicas:**
  * *Complejidad vs. Tamaño muestral:* Reflexión sobre por qué una red compleja (LSTM) no necesariamente aplasta a un árbol o MLP cuando se tienen alrededor de 250 datos mensuales.
  * *Validación en presencia de crisis:* La importancia de la validación secuencial para neutralizar distorsiones atípicas como la de 2020.

---

### Distribución de Tiempos Recomendada (para 20 min totales)
* **Parte 1 (Persona A):** 4:30 min
* **Parte 2 (Persona B):** 5:00 min
* **Parte 3 (Persona A):** 4:30 min
* **Parte 4 (Persona B):** 5:00 min
* **Preguntas / Discusión conjunta:** 1:00 min