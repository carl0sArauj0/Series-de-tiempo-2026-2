# ==============================================================================
# SCRIPT MAESTRO: ANÁLISIS DESCRIPTIVO DEL ISE (PASOS 1 AL 4) Y CREACIÓN DE HTML
# ==============================================================================

# 1. Lista de paquetes requeridos
paquetes <- c(
  "readxl", "tidyverse", "lubridate", "zoo", "xts", 
  "tsibble", "feasts", "fable", "timetk", "forecast", 
  "rmarkdown", "knitr", "plotly", "scales"
)

# Instalar los paquetes que falten
paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[, "Package"])]
if (length(paquetes_faltantes) > 0) {
  message("Instalando paquetes faltantes: ", paste(paquetes_faltantes, collapse = ", "))
  install.packages(paquetes_faltantes, dependencies = TRUE)
}

# Cargar rmarkdown
library(rmarkdown)

# 2. Generar dinámicamente el archivo R Markdown (.Rmd)
rmd_content <- '---
title: "Análisis Descriptivo de Series de Tiempo: Indicador de Seguimiento a la Economía (ISE)"
author: "Análisis Económico y Estadístico"
date: "`r Sys.Date()`"
output: 
  html_document:
    theme: flatly
    highlight: tango
    toc: true
    toc_float: true
    toc_depth: 3
    code_folding: show
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(
  echo = TRUE, 
  warning = FALSE, 
  message = FALSE, 
  fig.width = 10, 
  fig.height = 5.5,
  fig.align = "center"
)
library(readxl)
library(tidyverse)
library(lubridate)
library(zoo)
library(xts)
library(tsibble)
library(feasts)
library(fable)
library(timetk)
library(forecast)
library(plotly)
```

## 1. Naturaleza, Cálculo e Interpretación del ISE

### ¿Qué es el ISE?
El **Indicador de Seguimiento a la Economía (ISE)** es un índice sintético mensual calculado por el **DANE (Departamento Administrativo Nacional de Estadística)** de Colombia. Su objetivo fundamental es medir la evolución y dinámica de la actividad económica del país en el corto plazo, anticipando la tendencia del Producto Interno Bruto (PIB) trimestral.

### Metodología de Cálculo
El cálculo se fundamenta en las recomendaciones metodológicas del Sistema de Cuentas Nacionales (SCN 2008):
1. **Agrupación Sectorial:** Desagrega la economía en tres grandes ramas de actividad:
   * **Primarias:** Agricultura, ganadería, caza, silvicultura, pesca y explotación de minas y canteras.
   * **Secundarias:** Industrias manufactureras y construcción.
   * **Terciarias:** Comercio, transporte, alojamiento, servicios financieros, inmobiliarios, administración pública, educación y salud.
2. **Ponderación:** Cada rama se agrega a partir de indicadores mensuales de volumen físico (producción, ventas, nóminas, etc.) ponderados según su participación en el Valor Agregado Bruto del año base (actualmente año base 2015 = 100).
3. **Ajustes:** El DANE produce la serie en su versión *original* y *ajustada por efecto estacional y calendario*.

### Interpretación
* **Nivel del Índice:** Mide el volumen real de la actividad frente al periodo base (100).
* **Tasa de Crecimiento Anual:** $\left(\frac{\text{ISE}_t}{\text{ISE}_{t-12}} - 1\right) \times 100\%$, refleja el crecimiento o desaceleración económica interanual de dicho mes.

**Fuente oficial:** *Departamento Administrativo Nacional de Estadística (DANE) - Dirección de Síntesis y Cuentas Nacionales (DSCN), República de Colombia.*

---

## Paso 1: Configuración, Limpieza y Estructuración de Datos

Cargamos el archivo `ISE2005_2026.xlsx`, tratamos comas decimales y ordenamos cronológicamente.

```{r paso1}
# Lectura de la base de datos
datos_raw <- read_excel("ISE2005_2026.xlsx")

# Limpieza y estandarización de columnas
df_ise <- datos_raw %>%
  rename(Fecha_raw = 1, Iset = 2) %>%
  mutate(
    # Convertir a numérico por si contiene comas decimales
    Iset = as.numeric(gsub(",", ".", as.character(Iset))),
    # Parsear formato de fecha (Día/Mes/Año o Año-Mes-Día)
    Fecha = parse_date_time(Fecha_raw, orders = c("dmy", "ymd", "mdy")),
    Fecha = as.Date(Fecha)
  ) %>%
  filter(!is.na(Fecha), !is.na(Iset)) %>%
  arrange(Fecha) # Garantizar orden cronológico ascendente

head(df_ise)
tail(df_ise)

# Creación de objetos temporales
fecha_inicio <- c(year(min(df_ise$Fecha)), month(min(df_ise$Fecha)))

# 1. Objeto ts clásico
ise_ts <- ts(df_ise$Iset, start = fecha_inicio, frequency = 12)

# 2. Objeto xts
ise_xts <- xts(df_ise$Iset, order.by = as.yearmon(df_ise$Fecha), frequency = 12)

# 3. Objeto tsibble (ecosistema tidy)
ise_tsbl <- df_ise %>%
  mutate(Mes = yearmonth(Fecha)) %>%
  as_tsibble(index = Mes)

ise_ts
```

---

## Paso 2: Visualización Exploratoria Inicial

Analizamos visualmente la trayectoria del ISE para identificar tendencia, estacionalidad y choques exógenos.

```{r paso2}
# Gráfico interactivo con timetk
df_ise %>%
  plot_time_series(Fecha, Iset, 
                   .interactive = TRUE,
                   .title = "Evolución Mensual del ISE (Colombia)",
                   .y_lab = "Índice (Base 2015=100)",
                   .x_lab = "Año")
```

### Observaciones descriptivas iniciales:
* **Tendencia:** Crecimiento sostenido positivo a largo plazo impulsado por la expansión de la economía colombiana.
* **Choques estructurales:** Se observa una caída atípica abrupta en **abril de 2020** generada por los confinamientos del COVID-19, seguida por un rebote acelerado (2021-2022) y una fase posterior de estabilización.
* **Estacionalidad:** Patrones repetitivos intra-anuales (valles habituales en enero y picos consistentes hacia el último trimestre del año debido al ciclo comercial y festivo).

---

## Paso 3: Análisis y Estabilización de la Varianza Marginal

Verificamos si la oscilación estacional se amplifica conforme sube el nivel de la serie (heterocedasticidad marginal).

```{r paso3}
# Cálculo de Lambda óptimo de Box-Cox
lambda_loglik <- BoxCox.lambda(ise_ts, method = "loglik", lower = -1, upper = 3)
lambda_guerrero <- BoxCox.lambda(ise_ts, method = "guerrero", lower = -1, upper = 3)

cat("Lambda óptimo (Log-Likelihood):", round(lambda_loglik, 4), "\n")
cat("Lambda óptimo (Guerrero):", round(lambda_guerrero, 4), "\n")

# Selección del parámetro y transformación
lambda_opt <- lambda_guerrero
ise_boxcox <- BoxCox(ise_ts, lambda = lambda_opt)

# Comparación gráfica: Original vs. Transformada
par(mfrow = c(2,1), mar = c(3, 3, 2, 1))
plot(ise_ts, main = "Serie Original: ISE", col = "steelblue", lwd = 2, ylab = "")
plot(ise_boxcox, main = paste("Serie Transformada Box-Cox (Lambda =", round(lambda_opt, 2), ")"), 
     col = "darkgreen", lwd = 2, ylab = "")
par(mfrow = c(1,1))
```

*Interpretación:* La transformación Box-Cox homogeneiza la magnitud de las variaciones estacionales a lo largo del tiempo, facilitando la descomposición aditiva y los modelos lineales posteriores.

---

## Paso 4: Detección, Estimación y Eliminación de la Tendencia

Evaluamos la componente tendencial a través de distintos métodos tanto en la serie original como en la transformada.

### 4.1. Descomposición por Filtro de Promedios Móviles
```{r paso4_ma}
# Descomposición aditiva clásica
decomp_ise <- decompose(ise_ts, type = "additive")
plot(decomp_ise, col = "darkblue")

# Extracción de la tendencia mediante promedio móvil centrado (m = 12)
tendencia_ma <- forecast::ma(ise_ts, order = 12, centre = TRUE)

plot(ise_ts, main = "ISE y Estimación de Tendencia (Filtro MA 2x12)", col = "gray60", ylab = "ISE")
lines(tendencia_ma, col = "red", lwd = 2)
legend("topleft", legend = c("Serie Original", "Tendencia MA(12)"), 
       col = c("gray60", "red"), lty = 1, lwd = 2, bty = "n")
```

### 4.2. Tendencia Determinística y Serie Desestacionalizada por Regresión
```{r paso4_regresion}
# Ajuste de regresión lineal sobre el tiempo
tiempo <- time(ise_ts)
fit_tend <- lm(ise_ts ~ tiempo)
summary(fit_tend)

# Eliminación de tendencia determinística
ise_detrend_lm <- ise_ts - predict(fit_tend)

par(mfrow = c(2,1), mar = c(3, 3, 2, 1))
plot(ise_ts, main = "Tendencia Lineal Global", ylab = "")
abline(fit_tend, col = "firebrick", lwd = 2)

plot(ise_detrend_lm, main = "Serie sin Tendencia Lineal (Residuos)", col = "purple", ylab = "")
abline(h = 0, lty = 2, col = "gray")
par(mfrow = c(1,1))

# ACF de la serie sin tendencia lineal
acf(ise_detrend_lm, lag.max = 36, main = "ACF de la Serie Sin Tendencia Lineal")
```

*Nota:* El ACF de los residuos aún decae lentamente, lo que indica que la tendencia de la economía colombiana no es puramente lineal determinista, sino que contiene componentes no lineales o estocásticas.

### 4.3. Suavizamiento Robusto no Lineal (Descomposición STL)
```{r paso4_stl}
# Descomposición STL con tsibble y feasts
stl_fit <- ise_tsbl %>%
  model(STL(Iset ~ trend(window = 13) + season(window = "periodic"), robust = TRUE))

components(stl_fit) %>%
  autoplot() +
  labs(title = "Descomposición STL Robusta del ISE")
```

### 4.4. Tendencia Estocástica: Diferenciación Ordinaria
Si la serie posee una raíz unitaria (caminata aleatoria con deriva), la tendencia se elimina con la primera diferencia ordinaria: $\nabla Y_t = Y_t - Y_{t-1}$.

```{r paso4_diff}
diff_ise <- diff(ise_ts, differences = 1)
diff_boxcox <- diff(ise_boxcox, differences = 1)

par(mfrow = c(2,2), mar = c(3, 3, 2, 1))
plot(diff_ise, main = "Primera Diferencia (Serie Original)", col = "blue")
abline(h = 0, lty = 2, col = "red")

acf(diff_ise, lag.max = 36, main = "ACF: Primera Diferencia Original")

plot(diff_boxcox, main = "Primera Diferencia (Serie Box-Cox)", col = "darkgreen")
abline(h = 0, lty = 2, col = "red")

acf(diff_boxcox, lag.max = 36, main = "ACF: Primera Diferencia Box-Cox")
par(mfrow = c(1,1))
```

*Conclusión del Paso 4:* 
La aplicación de la primera diferencia ordinaria ($\nabla$) remueve con éxito la tendencia global, dejando expuesto un fuerte patrón estacional estocástico/determinístico con picos claros cada 12 meses en la función de autocorrelación (lags 12, 24, 36).
'

# 3. Escribir el archivo Rmd en el directorio
writeLines(rmd_content, "Analisis_Descriptivo_ISE.Rmd")
message("-> Archivo 'Analisis_Descriptivo_ISE.Rmd' generado exitosamente.")

# 4. Renderizar a HTML
message("-> Compilando reporte HTML interactivo...")
render(
  input = "Analisis_Descriptivo_ISE.Rmd", 
  output_file = "Analisis_Descriptivo_ISE.html"
)

message("====================================================================")
message("¡PROCESO FINALIZADO!")
message("Se ha creado el archivo HTML: ", file.path(getwd(), "Analisis_Descriptivo_ISE.html"))
message("Puedes abrirlo en cualquier navegador web.")
message("====================================================================")
```
