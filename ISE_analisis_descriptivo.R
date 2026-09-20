# ==============================================================================
# ANÁLISIS DESCRIPTIVO DE SERIES DE TIEMPO (PASOS 1 AL 4)
# SERIE ORIGINAL Y TRANSFORMADA: INDICADOR DE SEGUIMIENTO A LA ECONOMÍA (ISE)
# ==============================================================================
# Fuente: DANE (Departamento Administrativo Nacional de Estadística - Colombia)
# Metodología: Índice sintético mensual (Base 2015 = 100) que mide la evolución 
# de la actividad económica real en el corto plazo (ramas primarias, secundarias
# y terciarias). Una variación anual (ISE_t / ISE_{t-12} - 1) * 100 aproxima
# el crecimiento del PIB mensual.
# ==============================================================================

# 0. CARGA DE LIBRERÍAS --------------------------------------------------------
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

# ==============================================================================
# PASO 1: CONFIGURACIÓN, LIMPIEZA Y ESTRUCTURACIÓN DE DATOS
# ==============================================================================

# 1.1 Carga del archivo Excel
df_raw <- read_excel("Documents/Github/Series-de-tiempo-2026-2/ISE2005_2026.xlsx")

# 1.2 Limpieza, formato decimal y ordenamiento cronológico
df_ise <- df_raw %>%
  rename(Fecha_raw = 1, Iset = 2) %>%
  mutate(
    # Reemplazar comas por puntos en caso de leerse como texto
    Iset = as.numeric(gsub(",", ".", as.character(Iset))),
    # Identificar formato de fecha y convertir
    Fecha = as.Date(parse_date_time(Fecha_raw, orders = c("dmy", "ymd", "mdy")))
  ) %>%
  filter(!is.na(Fecha), !is.na(Iset)) %>%
  arrange(Fecha) # Garantizar orden cronológico ascendente

# 1.3 Creación de objetos temporales
fecha_inicio <- c(year(min(df_ise$Fecha)), month(min(df_ise$Fecha)))

# Serie Original (ts)
ise_ts <- ts(df_ise$Iset, start = fecha_inicio, frequency = 12)

# Objeto tsibble para la serie original
ise_tsbl <- df_ise %>%
  mutate(Mes = yearmonth(Fecha)) %>%
  as_tsibble(index = Mes)

# Inspección básica
cat("=== INFORMACIÓN DE LA SERIE ORIGINAL ===\n")
print(head(df_ise))
print(summary(ise_ts))


# ==============================================================================
# PASO 2: VISUALIZACIÓN EXPLORATORIA INICIAL
# ==============================================================================

# 2.1 Gráfico en serie temporal
plot(ise_ts, 
     main = "ISE Colombia (2005 - Presente) - Serie Original",
     ylab = "Índice (Base 2015 = 100)", 
     xlab = "Año", 
     col = "steelblue", 
     lwd = 2)
grid()

# 2.2 Gráfica interactiva / Tidy
df_ise %>%
  plot_time_series(Fecha, Iset, 
                   .interactive = FALSE, 
                   .title = "Evolución Mensual del ISE (Serie Original)",
                   .y_lab = "Índice Base 2015=100")

# Diagnóstico visual inicial:
# - Tendencia: Creciente a largo plazo.
# - Estacionalidad: Caídas en enero y picos marcados a final de año.
# - Quiebre estructural: Fuerte choque atípico en abril de 2020 (COVID-19).


# ==============================================================================
# PASO 3: ANÁLISIS Y ESTABILIZACIÓN DE LA VARIANZA MARGINAL (BOX-COX)
# ==============================================================================

# 3.1 Estimación del parámetro lambda óptimo
lambda_loglik   <- BoxCox.lambda(ise_ts, method = "loglik", lower = -1, upper = 3)
lambda_guerrero <- BoxCox.lambda(ise_ts, method = "guerrero", lower = -1, upper = 3)

cat("\n=== PARÁMETROS LAMBDA ESTIMADOS ===\n")
cat("Lambda (Log-Likelihood):", lambda_loglik, "\n")
cat("Lambda (Guerrero):", lambda_guerrero, "\n")

# Se selecciona el método de Guerrero para estabilizar varianza estacional
lambda_opt <- lambda_guerrero

# 3.2 Creación de la Serie Transformada Box-Cox
ise_boxcox_ts <- BoxCox(ise_ts, lambda = lambda_opt)

# Incorporar la serie transformada al tsibble
ise_tsbl <- ise_tsbl %>%
  mutate(Iset_bc = as.numeric(ise_boxcox_ts))

# 3.3 Comparación gráfica: Serie Original vs. Transformada
par(mfrow = c(2, 1), mar = c(3, 4, 2, 2))
plot(ise_ts, 
     main = "1. Serie Original: ISE", 
     ylab = "Nivel Original", col = "steelblue", lwd = 2)
grid()
plot(ise_boxcox_ts, 
     main = paste0("2. Serie Transformada Box-Cox (lambda = ", round(lambda_opt, 4), ")"), 
     ylab = "Nivel Transformado", col = "darkgreen", lwd = 2)
grid()
par(mfrow = c(1, 1))


# ==============================================================================
# PASO 4: DETECCIÓN, ESTIMACIÓN Y ELIMINACIÓN DE LA TENDENCIA
# (COMPARATIVO: SERIE ORIGINAL VS. SERIE TRANSFORMADA)
# ==============================================================================

# ------------------------------------------------------------------------------
# 4.1 Descomposición por Filtro de Promedios Móviles Centrados (m = 12)
# ------------------------------------------------------------------------------

# Estimación de la tendencia MA
tendencia_orig_ma <- forecast::ma(ise_ts, order = 12, centre = TRUE)
tendencia_bc_ma   <- forecast::ma(ise_boxcox_ts, order = 12, centre = TRUE)

# Eliminación de tendencia por sustracción (desestacionalizada / ciclo-estacional)
detrend_orig_ma <- ise_ts - tendencia_orig_ma
detrend_bc_ma   <- ise_boxcox_ts - tendencia_bc_ma

# Gráfica: Ajuste de la tendencia por Medias Móviles
par(mfrow = c(2, 1), mar = c(3, 4, 2, 2))
plot(ise_ts, main = "Tendencia MA(12) - Serie Original", col = "gray50", ylab = "Original")
lines(tendencia_orig_ma, col = "red", lwd = 2)
legend("topleft", legend = c("Original", "Tendencia MA(12)"), col = c("gray50", "red"), lty = 1, bty = "n")

plot(ise_boxcox_ts, main = "Tendencia MA(12) - Serie Transformada (Box-Cox)", col = "gray50", ylab = "Box-Cox")
lines(tendencia_bc_ma, col = "darkred", lwd = 2)
legend("topleft", legend = c("Transformada", "Tendencia MA(12)"), col = c("gray50", "darkred"), lty = 1, bty = "n")
par(mfrow = c(1, 1))

# Gráfica: Series sin tendencia por filtro MA
par(mfrow = c(2, 1), mar = c(3, 4, 2, 2))
plot(detrend_orig_ma, main = "Serie Original sin Tendencia (Residuos MA)", col = "steelblue", ylab = "")
abline(h = 0, lty = 2, col = "black")

plot(detrend_bc_ma, main = "Serie Transformada sin Tendencia (Residuos MA)", col = "darkgreen", ylab = "")
abline(h = 0, lty = 2, col = "black")
par(mfrow = c(1, 1))

# ------------------------------------------------------------------------------
# 4.2 Ajuste de Tendencia Determinística (Regresión Lineal)
# ------------------------------------------------------------------------------

tiempo <- time(ise_ts)

# Ajuste MCO
fit_orig_lm <- lm(ise_ts ~ tiempo)
fit_bc_lm   <- lm(ise_boxcox_ts ~ tiempo)

cat("\n=== RESUMEN REGRESIÓN: TENDENCIA LINEAL (ORIGINAL) ===\n")
print(summary(fit_orig_lm))

cat("\n=== RESUMEN REGRESIÓN: TENDENCIA LINEAL (TRANSFORMADA) ===\n")
print(summary(fit_bc_lm))

# Eliminación de la tendencia lineal (Residuos)
detrend_orig_lm <- ise_ts - predict(fit_orig_lm)
detrend_bc_lm   <- ise_boxcox_ts - predict(fit_bc_lm)

# Comparación gráfica de residuos sin tendencia lineal
par(mfrow = c(2, 2), mar = c(3, 4, 2, 2))
plot(detrend_orig_lm, main = "Sin Tendencia Lineal (Original)", col = "purple", ylab = "Residuos")
abline(h = 0, lty = 2)
acf(detrend_orig_lm, lag.max = 36, main = "ACF Sin Tendencia (Original)")

plot(detrend_bc_lm, main = "Sin Tendencia Lineal (Transformada)", col = "darkmagenta", ylab = "Residuos")
abline(h = 0, lty = 2)
acf(detrend_bc_lm, lag.max = 36, main = "ACF Sin Tendencia (Transformada)")
par(mfrow = c(1, 1))

# ------------------------------------------------------------------------------
# 4.3 Descomposición No Lineal Robusta (STL - Loess)
# ------------------------------------------------------------------------------

# Descomposición STL para la serie Original
stl_orig <- ise_tsbl %>%
  model(STL(Iset ~ trend(window = 13) + season(window = "periodic"), robust = TRUE))

# Descomposición STL para la serie Transformada
stl_bc <- ise_tsbl %>%
  model(STL(Iset_bc ~ trend(window = 13) + season(window = "periodic"), robust = TRUE))

# Visualización de componentes STL
components(stl_orig) %>% 
  autoplot() + 
  labs(title = "Descomposición STL Robusta: Serie Original ISE")

components(stl_bc) %>% 
  autoplot() + 
  labs(title = "Descomposición STL Robusta: Serie Transformada Box-Cox")

# ------------------------------------------------------------------------------
# 4.4 Tendencia Estocástica: Primera Diferencia Ordinaria (Nabla)
# ------------------------------------------------------------------------------

# Diferenciación ordinaria: (1 - B) Y_t = Y_t - Y_{t-1}
diff_orig <- diff(ise_ts, differences = 1)
diff_bc   <- diff(ise_boxcox_ts, differences = 1)

# Comparación: Series diferenciadas y sus respectivas funciones de autocorrelación (ACF)
par(mfrow = c(2, 2), mar = c(3, 4, 2, 2))
plot(diff_orig, main = "Primera Diferencia: Serie Original", col = "blue", ylab = "diff(Original)")
abline(h = 0, lty = 2, col = "red")
acf(diff_orig, lag.max = 36, main = "ACF: Primera Diferencia (Original)")

plot(diff_bc, main = "Primera Diferencia: Serie Transformada", col = "darkgreen", ylab = "diff(Box-Cox)")
abline(h = 0, lty = 2, col = "red")
acf(diff_bc, lag.max = 36, main = "ACF: Primera Diferencia (Box-Cox)")
par(mfrow = c(1, 1))

cat("\n=== ANÁLISIS DESCRIPTIVO (PASOS 1 AL 4) COMPLETADO ===\n")
