# ==============================================================================
# Análisis descriptivo del ISE, 2005-2026 — versión en script
#
# Es el mismo código de Analisis_ISE.qmd, organizado por secciones.
# Las explicaciones e interpretaciones están en el documento .qmd (y en el HTML).
#
# Cómo usarlo:
#   1. Abrir el proyecto con doble clic en ISE_Series_de_Tiempo.Rproj
#   2. Si es la primera vez en este computador, correr instalar_paquetes.R
#   3. Correr este script de arriba hacia abajo: Ctrl+Enter línea por línea,
#      o seleccionar una sección y Ctrl+Enter. El índice de secciones está
#      en el botón "Outline" (esquina superior derecha del editor).
# ==============================================================================


# 2. Librerías e importación de datos ------------------------------------------
## 2.1 Librerías -------------------------------------------------------------
## Si faltan paquetes, ejecutar primero el script instalar_paquetes.R (una sola vez)

library(readxl)        # lectura del archivo Excel
library(tidyverse)     # manipulación de datos y gráficos (dplyr, ggplot2, tidyr)
library(lubridate)     # manejo de fechas
library(zoo)           # clase yearmon para fechas mensuales
library(TSstudio)      # ts_info, ts_plot, ts_lags, ts_heatmap
library(forecast)      # BoxCox.lambda, ndiffs, nsdiffs, seasonaldummy, fourier
library(astsa)         # lag1.plot, mvspec
library(tsibble)       # objetos tsibble
library(feasts)        # STL, gg_subseries, gg_season, gg_tsresiduals
library(fable)         # TSLM para modelar la estacionalidad
library(fabletools)    # model, glance, tidy, augment, components
library(tseriesChaos)  # mutual: información mutua promedio (AMI)
library(biwavelet)     # transformada wavelet para explorar periodicidades
# here se usa como here::here() al importar los datos (no se carga con library()
# porque lubridate tiene una función con el mismo nombre en algunas versiones)

# MASS no se carga con library() porque su función select() oculta a
# dplyr::select(). Se usa directamente como MASS::boxcox().
# FitAR ya no está disponible en CRAN; su gráfico de log-verosimilitud
# de lambda se reemplaza por MASS::boxcox(), que hace lo mismo.

## 2.2 Importación -----------------------------------------------------------
ise <- read_excel(here::here("datos", "ISE2005_2026.xlsx"))
str(ise)
head(ise)
tail(ise)

## 2.3 Verificación de la base -----------------------------------------------
## Nombres de las columnas: la base ya trae "Fecha" e "Iset"; se dejan explícitos
names(ise)[1:2] <- c("Fecha", "Iset")

## Número de observaciones y datos faltantes
nrow(ise)
sum(is.na(ise$Iset))

## Rango de fechas
range(ise$Fecha)

## ¿Todas las fechas consecutivas están separadas exactamente por un mes?
## (si la base viniera en orden descendente, aquí aparecería -1 y habría que invertirla)
table(round(12 * diff(as.yearmon(ise$Fecha))))

## 2.4 Construcción de los objetos de serie de tiempo ------------------------
## Fechas en formato año-mes (se usan en la sección de Semana Santa y en el modelamiento)
Fechas_ise <- as.yearmon(ise$Fecha)

## Serie de tiempo mensual que empieza en enero de 2005
ise_ts <- ts(ise$Iset,
             start = c(year(min(ise$Fecha)), month(min(ise$Fecha))),
             frequency = 12)

ts_info(ise_ts)
class(ise_ts)


# 3. Gráfico de la serie -------------------------------------------------------
plot(ise_ts, main = "Índice de Seguimiento a la Economía (ISE)",
     ylab = "ISE (2015 = 100)", xlab = "Año")
## Periodo de las restricciones por COVID-19 (marzo a julio de 2020)
rect(2020 + 2/12, par("usr")[3], 2020 + 7/12, par("usr")[4],
     col = adjustcolor("red", alpha.f = 0.15), border = NA)

ts_plot(ise_ts,
        title = "Índice de Seguimiento a la Economía (ISE)",
        Ytitle = "ISE (2015 = 100)",
        Xtitle = "Año",
        Xgrid = TRUE,
        Ygrid = TRUE)

## Acercamiento a 2018-2022 para ver el patrón mensual y el choque de 2020
plot(window(ise_ts, start = c(2018, 1), end = c(2022, 12)), type = "o", pch = 20,
     main = "ISE 2018-2022", ylab = "ISE", xlab = "Año")
abline(v = 2018:2023, lty = 3, col = "grey60")


# 4. Estabilización de la varianza: transformación Box-Cox ---------------------
## 4.1 Evidencia gráfica: media y desviación estándar por año ----------------
## Se usan solo años completos (2026 tiene 6 meses)
anual <- tibble(anio = floor(time(ise_ts)), valor = as.numeric(ise_ts)) |>
  group_by(anio) |>
  filter(n() == 12) |>
  summarise(media = mean(valor), desv = sd(valor)) |>
  mutate(cv = 100 * desv / media,
         covid = anio %in% c(2020, 2021))

anual |> select(anio, media, desv, cv) |> mutate(across(-anio, ~ round(.x, 2))) |> print(n = Inf)

ggplot(anual, aes(media, desv)) +
  geom_smooth(data = filter(anual, !covid), method = "lm", se = FALSE,
              color = "grey50", linetype = "dashed") +
  geom_point(aes(color = covid), size = 2.5) +
  geom_text(aes(label = anio), vjust = -0.8, size = 3) +
  scale_color_manual(values = c("FALSE" = "black", "TRUE" = "red"),
                     labels = c("Resto de años", "2020-2021")) +
  labs(title = "Desviación estándar frente a media anual del ISE",
       x = "Media anual", y = "Desviación estándar anual", color = NULL) +
  theme_minimal()

## 4.2 Estimación de lambda --------------------------------------------------
## 1. Método de Guerrero con toda la muestra
lambda_guerrero <- forecast::BoxCox.lambda(ise_ts, method = "guerrero", lower = -1, upper = 3)

## 2. Método de Guerrero sin el periodo de la pandemia (2005-2019)
lambda_guerrero_pre <- forecast::BoxCox.lambda(window(ise_ts, end = c(2019, 12)),
                                               method = "guerrero", lower = -1, upper = 3)

## 3. Regla media-desviación: si log(desv) = a + b*log(media), entonces lambda = 1 - b
b_todos <- coef(lm(log(desv) ~ log(media), data = anual))[2]
b_sin_covid <- coef(lm(log(desv) ~ log(media), data = filter(anual, !covid)))[2]
lambda_md <- 1 - b_todos
lambda_md_sin_covid <- 1 - b_sin_covid

## 3. Log-verosimilitud perfil de lambda (reemplaza a FitAR::BoxCox).
## Busca el lambda con el que un modelo de regresión ajusta mejor. Se estima con
## dos formas de la tendencia:
##   (a) una recta + efectos mensuales
##   (b) un efecto por año + efectos mensuales (tendencia flexible)
datos_bc <- data.frame(y = as.numeric(ise_ts),
                       tiempo = as.numeric(time(ise_ts)),
                       anio = factor(floor(time(ise_ts))),
                       mes = factor(cycle(ise_ts)))

par(mfrow = c(1, 2))
bc_recta <- MASS::boxcox(y ~ tiempo + mes, data = datos_bc, lambda = seq(-1, 2, 0.01))
title("(a) Tendencia lineal")
bc_anio <- MASS::boxcox(y ~ anio + mes, data = datos_bc, lambda = seq(-1, 2, 0.01))
title("(b) Un efecto por año")
par(mfrow = c(1, 1))

lambda_mv <- bc_recta$x[which.max(bc_recta$y)]
lambda_mv_anio <- bc_anio$x[which.max(bc_anio$y)]

## Resumen de todas las estimaciones de lambda
tibble(Metodo = c("Guerrero, 2005-2026", "Guerrero, 2005-2019",
                  "Media-desviación, todos los años", "Media-desviación, sin 2020-2021",
                  "Verosimilitud perfil, tendencia lineal",
                  "Verosimilitud perfil, efecto por año"),
       lambda = round(c(lambda_guerrero, lambda_guerrero_pre, lambda_md,
                        lambda_md_sin_covid, lambda_mv, lambda_mv_anio), 2))

## 4.3 Decisión --------------------------------------------------------------
lise <- log(ise_ts)   # equivalente a forecast::BoxCox(ise_ts, lambda = 0)

par(mfrow = c(2, 1), mar = c(3, 4, 2.5, 1))
plot(ise_ts, main = "ISE sin transformar", ylab = "ISE", xlab = "")
plot(lise, main = "Logaritmo del ISE", ylab = "log(ISE)", xlab = "")
par(mfrow = c(1, 1))

## 4.4 Chequeo después de la transformación ----------------------------------
anual_log <- tibble(anio = floor(time(lise)), valor = as.numeric(lise)) |>
  group_by(anio) |> filter(n() == 12) |>
  summarise(media = mean(valor), desv = sd(valor)) |>
  mutate(covid = anio %in% c(2020, 2021))

ggplot(anual_log, aes(media, desv)) +
  geom_point(aes(color = covid), size = 2.5) +
  geom_text(aes(label = anio), vjust = -0.8, size = 3) +
  geom_smooth(data = filter(anual_log, !covid), method = "lm", se = FALSE,
              color = "grey50", linetype = "dashed") +
  scale_color_manual(values = c("FALSE" = "black", "TRUE" = "red"), guide = "none") +
  labs(title = "Desviación estándar frente a media anual del log(ISE)",
       x = "Media anual de log(ISE)", y = "Desviación estándar anual") +
  theme_minimal()

## Pendiente de la relación sin 2020-2021 (cercana a cero indica varianza estabilizada)
chk_log <- lm(desv ~ media, data = filter(anual_log, !covid))
summary(chk_log)$coefficients

## Cambio de la desviación estándar ajustada entre el año de menor y el de mayor nivel,
## antes y después de transformar (sin 2020-2021)
chk_nivel <- lm(desv ~ media, data = filter(anual, !covid))
rango_nivel <- range(filter(anual, !covid)$media)
rango_log <- range(filter(anual_log, !covid)$media)
cambio_nivel <- 100 * diff(predict(chk_nivel, data.frame(media = rango_nivel))) /
  predict(chk_nivel, data.frame(media = rango_nivel[1]))
cambio_log <- 100 * diff(predict(chk_log, data.frame(media = rango_log))) /
  predict(chk_log, data.frame(media = rango_log[1]))
round(c(cambio_nivel = unname(cambio_nivel), cambio_log = unname(cambio_log)), 1)


# 5. Análisis de tendencia -----------------------------------------------------
## 5.1 Tendencia determinística lineal ---------------------------------------
fit_ise <- lm(lise ~ time(lise), na.action = NULL)
summary(fit_ise)

crec_anual <- 100 * (exp(coef(fit_ise)[2]) - 1)
crec_anual

plot(lise, ylab = "log(ISE)", main = "log(ISE) con recta de tendencia")
abline(fit_ise, col = "red", lwd = 2)

## Serie sin tendencia determinística (residuales de la regresión)
ise_sin_tend <- lise - predict(fit_ise)
plot(ise_sin_tend, main = "log(ISE) sin tendencia lineal (residuales)", ylab = "Residual")
abline(h = 0, lty = 2)

## 5.2 Tendencias polinómicas: cuadrática y cúbica ---------------------------
t_anios <- as.numeric(time(lise)) - 2005

fit_lin  <- lm(lise ~ t_anios)
fit_cuad <- lm(lise ~ t_anios + I(t_anios^2))
fit_cub  <- lm(lise ~ t_anios + I(t_anios^2) + I(t_anios^3))

summary(fit_cub)

## Comparación de las tres tendencias
comparar_tend <- function(m) {
  r <- resid(m)
  c(R2_ajustado = summary(m)$adj.r.squared,
    AIC = AIC(m), BIC = BIC(m),
    ACF_resid_lag1 = acf(r, plot = FALSE)$acf[2],
    ACF_resid_lag12 = acf(r, plot = FALSE)$acf[13])
}
tabla_tend <- rbind(Lineal = comparar_tend(fit_lin),
                    Cuadratica = comparar_tend(fit_cuad),
                    Cubica = comparar_tend(fit_cub))
round(tabla_tend, 3)

plot(lise, ylab = "log(ISE)", main = "log(ISE) con tendencias lineal, cuadrática y cúbica", col = "grey40")
lines(ts(fitted(fit_lin),  start = start(lise), frequency = 12), col = "red",       lwd = 2)
lines(ts(fitted(fit_cuad), start = start(lise), frequency = 12), col = "darkgreen", lwd = 2, lty = 2)
lines(ts(fitted(fit_cub),  start = start(lise), frequency = 12), col = "blue",      lwd = 2)
legend("topleft", c("Lineal", "Cuadrática", "Cúbica"), col = c("red", "darkgreen", "blue"),
       lty = c(1, 2, 1), lwd = 2, bty = "n")

## Crecimiento anual implícito en la tendencia cúbica: la derivada de la tendencia
## respecto a t es la tasa de crecimiento instantánea del ISE
b <- coef(fit_cub)
crec_cub <- function(t) 100 * (exp(b[2] + 2 * b[3] * t + 3 * b[4] * t^2) - 1)

anios_graf <- seq(0, max(t_anios), by = 1 / 12)
plot(2005 + anios_graf, crec_cub(anios_graf), type = "l", lwd = 2, col = "blue",
     main = "Crecimiento anual implícito en la tendencia cúbica",
     xlab = "Año", ylab = "% anual")
abline(h = crec_anual, lty = 2, col = "red")
legend("topright", c("Tendencia cúbica", "Tendencia lineal (constante)"),
       col = c("blue", "red"), lty = c(1, 2), lwd = 2, bty = "n")

## Año en que el crecimiento implícito llega a su mínimo
anio_min <- 2005 + anios_graf[which.min(crec_cub(anios_graf))]
round(c(crec_2005 = crec_cub(0), crec_2015 = crec_cub(10),
        crec_minimo = min(crec_cub(anios_graf)), anio_minimo = anio_min,
        crec_2026 = crec_cub(max(t_anios))), 2)

## Residuales de la tendencia cúbica (se usan en la sección de diferenciación)
ise_sin_tend_cub <- lise - fitted(fit_cub)

## 5.3 Suavizamientos no paramétricos ----------------------------------------
## Promedio móvil centrado 2x12: pondera con 1/24 los extremos y 1/12 los demás.
## Al cubrir exactamente un año, elimina la estacionalidad y deja la tendencia-ciclo.
wgts <- c(.5, rep(1, 11), .5) / 12
lise_ma <- stats::filter(lise, sides = 2, filter = wgts)

plot(lise, main = "Suavizamiento por promedio móvil 2x12", ylab = "log(ISE)")
lines(lise_ma, lwd = 2, col = 4)

## Kernel gaussiano (Nadaraya-Watson) con ancho de banda de 1 año
plot(lise, main = "Suavizamiento kernel (Nadaraya-Watson)", ylab = "log(ISE)")
lines(ksmooth(time(lise), lise, "normal", bandwidth = 1), lwd = 2, col = 4)

## Lowess (regresión local por vecinos más cercanos). El parámetro f es la fracción
## de observaciones usada en cada ajuste local; se fija para usar 12, 24 y 36
## vecinos, es decir, uno, dos y tres años de datos.
n_lise <- length(lise)
f_12 <- 12 / n_lise
f_24 <- 24 / n_lise
f_36 <- 36 / n_lise

plot(lise, main = "Suavizamiento Lowess con 12, 24 y 36 vecinos", ylab = "log(ISE)", col = "grey50")
lines(lowess(lise, f = f_12), lwd = 2, col = "orange")
lines(lowess(lise, f = f_24), lwd = 2, col = "blue")
lines(lowess(lise, f = f_36), lwd = 2, col = "red")
legend("topleft", c("12 vecinos (1 año)", "24 vecinos (2 años)", "36 vecinos (3 años)"),
       col = c("orange", "blue", "red"), lwd = 2, bty = "n")

plot(lise, main = "Suavizamiento por splines", ylab = "log(ISE)")
lines(smooth.spline(time(lise), lise, spar = .5), lwd = 2, col = 4)       # menos suave
lines(smooth.spline(time(lise), lise, spar = 1), lty = 2, lwd = 2, col = 2) # más suave
legend("topleft", c("spar = 0.5", "spar = 1"), col = c(4, 2), lty = c(1, 2), lwd = 2, bty = "n")

## Loess con ggplot2 (reemplaza timetk::smooth_vec, con el mismo span y grado)
df_ise <- tibble(Fecha = as.Date(Fechas_ise), lISE = as.numeric(lise))

ggplot(df_ise, aes(Fecha, lISE)) +
  geom_line(color = "grey40") +
  geom_smooth(method = "loess", span = 0.5, method.args = list(degree = 1),
              se = FALSE, color = "red") +
  labs(title = "Tendencia Loess sobre log(ISE)", x = "Año", y = "log(ISE)") +
  theme_minimal()

## 5.4 Descomposición --------------------------------------------------------
## Descomposición clásica aditiva sobre el logaritmo
## (equivale a una descomposición multiplicativa sobre el ISE original)
ise_decompo <- decompose(lise)
plot(ise_decompo)

## Descomposición STL robusta (Loess), con estacionalidad fija
lise_tsbl <- as_tsibble(lise)

lise_tsbl |>
  model(STL(value ~ trend() + season(window = "periodic"), robust = TRUE)) |>
  components() |>
  autoplot() +
  labs(title = "Descomposición STL de log(ISE)")

## Fuerza de la tendencia y de la estacionalidad (entre 0 y 1)
lise_tsbl |> features(value, feat_stl) |> select(trend_strength, seasonal_strength_year)

## 5.5 Diferenciación --------------------------------------------------------
dlise <- diff(lise)

par(mfrow = c(2, 1), mar = c(3, 4, 2.5, 1))
plot(ise_sin_tend_cub, type = "l", main = "Sin tendencia (residuales de la tendencia cúbica)", ylab = "", xlab = "")
abline(h = 0, lty = 2)
plot(dlise, type = "l", main = "Primera diferencia de log(ISE)", ylab = "", xlab = "")
abline(h = 0, lty = 2)
par(mfrow = c(1, 1))

## En el eje de retardos de acf() para una serie mensual, 1 = 12 meses
par(mfrow = c(1, 3), mar = c(4, 4, 3, 1))
acf(lise, 48, main = "ACF log(ISE)")
acf(ise_sin_tend_cub, 48, main = "ACF sin tendencia cúbica")
acf(dlise, 48, main = "ACF primera diferencia")
par(mfrow = c(1, 1))

## Número de diferencias ordinarias sugeridas por pruebas de raíz unitaria
forecast::ndiffs(lise, test = "kpss")
forecast::ndiffs(lise, test = "adf")


# 6. Análisis de dependencia ---------------------------------------------------
## 6.1 Diagramas de dispersión de retardos -----------------------------------
## Cada panel es Z_t frente a Z_{t-h} para h = 1,...,12, con su correlación
## y un suavizamiento lowess para detectar relaciones no lineales
par(mar = c(3, 2, 3, 2))
astsa::lag1.plot(dlise, 12)

TSstudio::ts_lags(dlise, lags = 1:12)

## 6.2 Función de autocorrelación simple y parcial ---------------------------
par(mfrow = c(2, 1), mar = c(4, 4, 2.7, 1))
acf(dlise, 48, main = "ACF de la primera diferencia de log(ISE)")
pacf(dlise, 48, main = "PACF de la primera diferencia de log(ISE)")
par(mfrow = c(1, 1))

## Prueba de Ljung-Box: H0 = no hay autocorrelación hasta el retardo 24
Box.test(dlise, lag = 24, type = "Ljung-Box")

## 6.3 Información mutua promedio (AMI) --------------------------------------
## 16 particiones: aproximadamente la raíz cuadrada del número de datos (257)
ami_ise <- tseriesChaos::mutual(as.numeric(dlise), partitions = 16, lag.max = 36, plot = FALSE)

plot(0:36, as.numeric(ami_ise), type = "h", lwd = 2,
     main = "Información mutua promedio de la primera diferencia de log(ISE)",
     xlab = "Retardo (meses)", ylab = "AMI")
abline(v = c(12, 24, 36), lty = 3, col = "red")

## 6.4 Correlación de distancia ----------------------------------------------
## Correlación de distancia de Székely, Rizzo y Bakirov (2007), calculada en R base.
## Da el mismo resultado que energy::dcor() sin depender de ese paquete.
dcor_base <- function(x, y) {
  A <- as.matrix(dist(x))                                  # distancias entre todos los pares de x
  B <- as.matrix(dist(y))
  A <- A - outer(rowMeans(A), colMeans(A), "+") + mean(A)  # doble centrado
  B <- B - outer(rowMeans(B), colMeans(B), "+") + mean(B)
  sqrt(mean(A * B) / sqrt(mean(A * A) * mean(B * B)))
}

calcular_dacf_interna <- function(serie, max_lag = 20) {
  sapply(1:max_lag, function(k) {
    n_k <- length(serie)
    x_t  <- serie[(k + 1):n_k]      # Z_t
    x_tk <- serie[1:(n_k - k)]      # Z_{t-k}
    dcor_base(x_t, x_tk)
  })
}

z <- as.numeric(dlise)
lags <- 1:36
d_acf_vals <- calcular_dacf_interna(z, max_lag = 36)

## Valor crítico aproximado: dCor entre pares de una serie permutada (independencia)
set.seed(2026)
dcor_nulo <- replicate(500, {
  zp <- sample(z)
  dcor_base(zp[-1], zp[-length(zp)])
})
umbral_dcor <- quantile(dcor_nulo, 0.95)

## Comparación con el valor absoluto del ACF
acf_vals <- acf(z, lag.max = 36, plot = FALSE)$acf[-1]

plot(lags, d_acf_vals, type = "h", lwd = 3, col = "darkgreen", ylim = c(0, 1.15),
     main = "Correlación de distancia (dCor) frente a |ACF|",
     xlab = "Retardo (meses)", ylab = "")
points(lags + 0.3, abs(acf_vals), type = "h", lwd = 2, col = "grey50")
abline(h = umbral_dcor, lty = 2, col = "blue")
legend("top", c("dCor", "|ACF|", "Percentil 95 bajo independencia"), horiz = TRUE,
       col = c("darkgreen", "grey50", "blue"), lty = c(1, 1, 2), lwd = c(3, 2, 1), bty = "n", cex = 0.85)

## Serie sin las medias mensuales: se resta a cada observación el promedio de su mes
z_sm <- as.numeric(resid(lm(z ~ factor(cycle(dlise)))))

## Misma serie con marzo-julio de 2020 neutralizados (fijados en cero, su media)
fechas_d <- as.yearmon(time(dlise))
covid_idx <- which(fechas_d >= as.yearmon(2020 + 2/12) & fechas_d <= as.yearmon(2020 + 6/12))
z_sm_nc <- z_sm
z_sm_nc[covid_idx] <- 0

umbral_perm <- function(serie, B = 500) {
  quantile(replicate(B, {sp <- sample(serie); dcor_base(sp[-1], sp[-length(sp)])}), 0.95)
}

set.seed(2026)
comparar <- list(
  "Sin medias mensuales" = z_sm,
  "Sin medias mensuales ni pandemia" = z_sm_nc
)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
for (nombre in names(comparar)) {
  serie <- comparar[[nombre]]
  dc <- calcular_dacf_interna(serie, max_lag = 24)
  ac <- abs(acf(serie, lag.max = 24, plot = FALSE)$acf[-1])
  plot(1:24, dc, type = "h", lwd = 3, col = "darkgreen", ylim = c(0, 0.5),
       main = nombre, xlab = "Retardo (meses)", ylab = "")
  points(1:24 + 0.3, ac, type = "h", lwd = 2, col = "grey50")
  abline(h = umbral_perm(serie), lty = 2, col = "blue")
}
par(mfrow = c(1, 1))

## Retardos 1, 2, 3 y 12
dcor_tabla <- tibble(
  retardo = c(1, 2, 3, 12),
  dcor_sin_medias = calcular_dacf_interna(z_sm, 12)[c(1, 2, 3, 12)],
  acf_sin_medias = acf(z_sm, 12, plot = FALSE)$acf[c(2, 3, 4, 13)],
  dcor_sin_pandemia = calcular_dacf_interna(z_sm_nc, 12)[c(1, 2, 3, 12)],
  acf_sin_pandemia = acf(z_sm_nc, 12, plot = FALSE)$acf[c(2, 3, 4, 13)]
)
dcor_tabla |> mutate(across(-retardo, ~ round(.x, 3)))


# 7. Detección de la estacionalidad --------------------------------------------
## 7.1 Por qué no se usa la serie original -----------------------------------
per_nivel <- astsa::mvspec(ise_ts, log = "no", plot = FALSE)
1 / per_nivel$freq[which.max(per_nivel$spec)] * 12   # periodo en meses del pico

## 7.2 Mapa de calor, monthplot y subseries ----------------------------------
TSstudio::ts_heatmap(dlise, title = "Mapa de calor: crecimiento mensual del ISE (dif. de log)")

monthplot(dlise, main = "Monthplot del crecimiento mensual del ISE", ylab = "dif. log(ISE)")
abline(h = 0, lty = 2)

dlise_tsbl <- as_tsibble(dlise)

dlise_tsbl |>
  gg_subseries(value) +
  labs(title = "Subseries mensuales del crecimiento del ISE", y = "dif. log(ISE)")

## Patrón de cada año sobre el log(ISE) en nivel
lise_tsbl |>
  gg_season(value, labels = "right") +
  labs(title = "Gráfico estacional de log(ISE) por año", y = "log(ISE)")

## 7.3 Boxplots y medidas descriptivas por mes -------------------------------
dlise_df <- tibble(anio = floor(time(dlise)),
                   mes = factor(month.abb[cycle(dlise)], levels = month.abb),
                   crec = 100 * as.numeric(dlise))

ggplot(dlise_df, aes(mes, crec)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Crecimiento mensual del ISE por mes (%)",
       x = "Mes", y = "100 x dif. log(ISE)") +
  theme_minimal()

ise_summary <- dlise_df |>
  group_by(mes) |>
  summarise(media = mean(crec), mediana = median(crec), sd = sd(crec),
            min = min(crec), max = max(crec)) |>
  mutate(across(-mes, ~ round(.x, 2)))
ise_summary

## 7.4 Periodograma ----------------------------------------------------------
## Con un objeto ts de frecuencia 12, mvspec reporta la frecuencia en ciclos por año
ise.per <- astsa::mvspec(dlise, log = "no", main = "Periodograma de la primera diferencia de log(ISE)")
abline(v = 1:6, lty = 2, col = "blue")

## Los cinco picos más altos
picos <- order(ise.per$spec, decreasing = TRUE)[1:5]
tibble(frecuencia_ciclos_por_anio = round(ise.per$freq[picos], 3),
       periodo_meses = round(12 / ise.per$freq[picos], 2),
       densidad = round(ise.per$spec[picos], 5))

## El mismo periodograma calculado directamente con la transformada rápida de Fourier.
## Aquí la frecuencia está en ciclos por mes (1/12 = 1 ciclo al año).
n_ise <- length(dlise)
Px_ise <- Mod(fft(as.numeric(dlise) - mean(dlise)))^2 / n_ise
Freq_ise <- (0:(n_ise - 1)) / n_ise

plot(Freq_ise[2:(n_ise %/% 2)], Px_ise[2:(n_ise %/% 2)], type = "h",
     main = "Periodograma vía transformada de Fourier",
     xlab = "Frecuencia (ciclos por mes)", ylab = "Periodograma")
abline(v = (1:6) / 12, lty = 2, col = "blue")

u_ise <- which.max(Px_ise[2:(n_ise %/% 2)]) + 1
sprintf("La frecuencia donde se maximiza el periodograma (FFT) es %s ciclos por mes", round(Freq_ise[u_ise], 4))
sprintf("El periodo asociado es aproximadamente: %s meses", round(1 / Freq_ise[u_ise], 2))

astsa::mvspec(dlise, spans = c(3, 3), log = "no",
              main = "Periodograma suavizado (Daniell modificado)")
abline(v = 1:6, lty = 2, col = "blue")

## 7.5 Wavelets --------------------------------------------------------------
## El tiempo se expresa en años, así que el periodo también queda en años
## (0,25 = 3 meses; 0,5 = 6 meses; 1 = 12 meses)
series_wavelets <- cbind(as.numeric(time(dlise)), as.numeric(dlise))
wt_ise <- biwavelet::wt(series_wavelets)

par(oma = c(0, 0, 0, 1), mar = c(5, 4, 4, 5) + 0.1)
plot(wt_ise, type = "power.corr.norm", plot.cb = TRUE,
     main = "Espectro wavelet (corregido por sesgo) del crecimiento mensual del ISE",
     xlab = "Año", ylab = "Periodo (años)")

## 7.6 Efecto de Semana Santa ------------------------------------------------
## Fecha del domingo de Pascua (algoritmo gregoriano de Meeus/Jones/Butcher)
fecha_pascua <- function(anio) {
  a <- anio %% 19; b <- anio %/% 100; c <- anio %% 100
  d <- b %/% 4; e <- b %% 4; f <- (b + 8) %/% 25; g <- (b - f + 1) %/% 3
  h <- (19 * a + b - d - g + 15) %% 30; i <- c %/% 4; k <- c %% 4
  l <- (32 + 2 * e + 2 * i - h - k) %% 7; m <- (a + 11 * h + 22 * l) %/% 451
  mes <- (h + l - 7 * m + 114) %/% 31; dia <- ((h + l - 7 * m + 114) %% 31) + 1
  as.Date(sprintf("%d-%02d-%02d", anio, mes, dia))
}

anios <- 2005:2026
jueves_santo <- fecha_pascua(anios) - 3
viernes_santo <- fecha_pascua(anios) - 2
festivos_ss <- c(jueves_santo, viernes_santo)

## Proporción de los dos festivos de Semana Santa que cae en cada mes (0, 0.5 o 1)
ss_nivel <- sapply(Fechas_ise, function(fm) sum(as.yearmon(festivos_ss) == fm) / 2)

## Años en que los festivos cayeron en marzo
anios_ss_marzo <- anios[month(viernes_santo) == 3]
anios_ss_marzo

## Crecimiento promedio de marzo y abril según el mes de Semana Santa (sin 2020)
dlise_df |>
  filter(mes %in% c("Mar", "Apr"), anio != 2020) |>
  mutate(semana_santa = if_else(anio %in% anios_ss_marzo, "En marzo", "En abril")) |>
  group_by(mes, semana_santa) |>
  summarise(crec_promedio = round(mean(crec), 2), n_anios = n(), .groups = "drop")

dlise_df |>
  filter(mes %in% c("Mar", "Apr"), anio != 2020) |>
  mutate(semana_santa = if_else(anio %in% anios_ss_marzo, "Semana Santa en marzo", "Semana Santa en abril")) |>
  ggplot(aes(mes, crec, color = semana_santa)) +
  geom_jitter(width = 0.15, size = 2.5) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Crecimiento de marzo y abril según el mes de Semana Santa",
       x = NULL, y = "100 x dif. log(ISE)", color = NULL) +
  theme_minimal()

## 7.7 Estacionalidad estocástica --------------------------------------------
## Número de diferencias estacionales sugeridas (prueba basada en la fuerza estacional)
forecast::nsdiffs(lise)

par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
acf(dlise, 48, main = "ACF dif. ordinaria")
acf(diff(dlise, lag = 12), 48, main = "ACF dif. ordinaria y estacional")
par(mfrow = c(1, 1))


# 8. Modelamiento de la estacionalidad -----------------------------------------
## 8.1 Variables dummy y armónicos -------------------------------------------
## Primeras filas de la matriz de dummies estacionales (enero es la categoría base)
head(forecast::seasonaldummy(dlise), 13)

## Primeras filas de los términos de Fourier con K = 2
head(forecast::fourier(dlise, K = 2), 13)

harmonics <- forecast::fourier(dlise, K = 6)
par(mar = c(1, 4, 1, 1), mfrow = c(6, 2))
for (i in 1:ncol(harmonics)) {
  plot(window(ts(harmonics[, i], start = start(dlise), frequency = 12), end = c(2007, 12)),
       type = "l", xlab = "", ylab = colnames(harmonics)[i])
}
par(mar = c(5, 4, 4, 2) + 0.1, mfrow = c(1, 1))

## 8.2 Regresores de calendario y de la pandemia -----------------------------
datos_mod <- tibble(
  index = yearmonth(as.Date(Fechas_ise[-1])),
  dlog  = as.numeric(dlise),
  d_ss  = diff(ss_nivel)
) |>
  mutate(covid_mar = as.integer(index == yearmonth("2020 Mar")),
         covid_abr = as.integer(index == yearmonth("2020 Apr")),
         covid_may = as.integer(index == yearmonth("2020 May")),
         covid_jun = as.integer(index == yearmonth("2020 Jun")),
         covid_jul = as.integer(index == yearmonth("2020 Jul"))) |>
  as_tsibble(index = index)

datos_mod

## 8.3 Ajuste y comparación de modelos ---------------------------------------
modelos <- datos_mod |>
  model(
    Fourier_K1 = TSLM(dlog ~ fourier(K = 1)),
    Fourier_K2 = TSLM(dlog ~ fourier(K = 2)),
    Fourier_K3 = TSLM(dlog ~ fourier(K = 3)),
    Fourier_K4 = TSLM(dlog ~ fourier(K = 4)),
    Fourier_K5 = TSLM(dlog ~ fourier(K = 5)),
    Fourier_K6 = TSLM(dlog ~ fourier(K = 6)),
    Dummy      = TSLM(dlog ~ season()),
    Dummy_SS   = TSLM(dlog ~ season() + d_ss),
    Dummy_SS_COVID = TSLM(dlog ~ season() + d_ss +
                            covid_mar + covid_abr + covid_may + covid_jun + covid_jul)
  )

comparacion <- glance(modelos) |>
  select(.model, df, r_squared, adj_r_squared, sigma2, AIC, BIC) |>
  mutate(sigma = sqrt(sigma2)) |>
  select(-sigma2) |>
  arrange(BIC)

comparacion |> mutate(across(where(is.numeric), ~ round(.x, 4)))

comparacion |>
  filter(str_detect(.model, "Fourier")) |>
  mutate(K = as.integer(str_extract(.model, "\\d"))) |>
  ggplot(aes(K, adj_r_squared)) +
  geom_line() + geom_point(size = 2.5) +
  geom_hline(yintercept = comparacion$adj_r_squared[comparacion$.model == "Dummy"],
             linetype = "dashed", color = "red") +
  annotate("text", x = 1.5, y = comparacion$adj_r_squared[comparacion$.model == "Dummy"] + 0.04,
           label = "Dummies", color = "red") +
  scale_x_continuous(breaks = 1:6) +
  labs(title = "R² ajustado según el número de armónicos de Fourier",
       x = "K (pares seno-coseno)", y = "R² ajustado") +
  theme_minimal()

## 8.4 Ajustados frente a observados -----------------------------------------
augment(modelos) |>
  filter(.model %in% c("Fourier_K2", "Fourier_K4", "Dummy_SS_COVID")) |>
  ggplot(aes(x = index)) +
  geom_line(aes(y = dlog), color = "black") +
  geom_line(aes(y = .fitted, color = .model), linewidth = 0.7) +
  facet_wrap(~ .model, ncol = 1) +
  labs(title = "Crecimiento mensual observado (negro) y ajustado",
       x = NULL, y = "dif. log(ISE)", color = NULL) +
  theme_minimal() + theme(legend.position = "none")

## Acercamiento 2015-2019 para comparar la forma del patrón estacional
augment(modelos) |>
  filter(.model %in% c("Fourier_K2", "Fourier_K4", "Dummy_SS_COVID"),
         index >= yearmonth("2015 Jan"), index <= yearmonth("2019 Dec")) |>
  ggplot(aes(x = index)) +
  geom_line(aes(y = dlog), color = "black", linewidth = 0.8) +
  geom_line(aes(y = .fitted, color = .model), linewidth = 0.7) +
  labs(title = "Observado frente a ajustado, 2015-2019",
       x = NULL, y = "dif. log(ISE)", color = "Modelo") +
  theme_minimal()

## 8.5 Efectos estacionales estimados ----------------------------------------
modelo_final <- modelos |> select(Dummy_SS_COVID)
coefs <- tidy(modelo_final)
coefs |> mutate(across(where(is.numeric), ~ round(.x, 4)))

## Crecimiento esperado de cada mes = intercepto (enero) + efecto del mes + efecto Semana Santa.
## Se calcula para los dos escenarios posibles de Semana Santa, sin pandemia.
b0 <- coefs$estimate[coefs$term == "(Intercept)"]
b_ss <- coefs$estimate[coefs$term == "d_ss"]

efectos <- coefs |>
  filter(str_detect(term, "season")) |>
  mutate(mes_num = as.integer(str_extract(term, "\\d+$"))) |>
  select(mes_num, estimate) |>
  add_row(mes_num = 1, estimate = 0) |>
  arrange(mes_num)

escenarios <- bind_rows(
  efectos |> mutate(escenario = "Semana Santa en abril",
                    d_ss = case_when(mes_num == 4 ~ 1, mes_num == 5 ~ -1, TRUE ~ 0)),
  efectos |> mutate(escenario = "Semana Santa en marzo",
                    d_ss = case_when(mes_num == 3 ~ 1, mes_num == 4 ~ -1, TRUE ~ 0))
) |>
  mutate(mes = factor(month.abb[mes_num], levels = month.abb),
         crec_esperado = 100 * (b0 + estimate + b_ss * d_ss))

ggplot(escenarios, aes(mes, crec_esperado, fill = escenario)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.75) +
  geom_hline(yintercept = 0) +
  scale_fill_manual(values = c("Semana Santa en abril" = "steelblue",
                               "Semana Santa en marzo" = "darkorange")) +
  labs(title = "Crecimiento mensual esperado según el modelo Dummy_SS_COVID",
       x = NULL, y = "%", fill = NULL) +
  theme_minimal() + theme(legend.position = "top")

escenarios |>
  select(mes, escenario, crec_esperado) |>
  pivot_wider(names_from = escenario, values_from = crec_esperado) |>
  mutate(across(-mes, ~ round(.x, 2)))

## 8.6 Diagnóstico de los residuales -----------------------------------------
modelo_final |> gg_tsresiduals(lag_max = 36) +
  labs(title = "Residuales del modelo Dummy_SS_COVID")

augment(modelo_final) |> features(.innov, ljung_box, lag = 24)

## 8.7 Desestacionalización --------------------------------------------------
stl_ise <- lise_tsbl |>
  model(STL(value ~ trend() + season(window = 13), robust = TRUE)) |>
  components()

ise_desest <- stl_ise |>
  mutate(Original = exp(value),
         Desestacionalizada = exp(season_adjust),
         Tendencia = exp(trend)) |>
  as_tibble() |>
  select(index, Original, Desestacionalizada, Tendencia) |>
  pivot_longer(-index, names_to = "serie", values_to = "valor")

ggplot(ise_desest, aes(as.Date(index), valor, color = serie)) +
  geom_line(aes(linewidth = serie)) +
  scale_color_manual(values = c(Original = "grey70", Desestacionalizada = "black", Tendencia = "red")) +
  scale_linewidth_manual(values = c(Original = 0.5, Desestacionalizada = 0.8, Tendencia = 0.8), guide = "none") +
  labs(title = "ISE original, desestacionalizado y tendencia (STL)",
       x = NULL, y = "ISE (2015 = 100)", color = NULL) +
  theme_minimal() + theme(legend.position = "top")

## Factor estacional promedio por mes: cuánto se desvía cada mes del nivel desestacionalizado
stl_ise |>
  as_tibble() |>
  mutate(mes = factor(month.abb[month(index)], levels = month.abb)) |>
  group_by(mes) |>
  summarise(factor_estacional_pct = round(100 * (exp(mean(season_year)) - 1), 2))
