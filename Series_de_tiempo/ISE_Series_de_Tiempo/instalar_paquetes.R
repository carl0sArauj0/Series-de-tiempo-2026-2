# Instala los paquetes del análisis del ISE.
# Correr una sola vez con el botón "Source".

paquetes <- c("here", "rmarkdown", "knitr", "readxl", "tidyverse", "lubridate",
              "zoo", "xts", "TSstudio", "forecast", "astsa", "tsibble", "feasts",
              "fable", "fabletools", "tseriesChaos", "biwavelet", "MASS")

# Instala los que no estén o no carguen, sin compilar (evita la pregunta y Rtools)
options(install.packages.compile.from.source = "never")
faltan <- paquetes[!sapply(paquetes, requireNamespace, quietly = TRUE)]
if (length(faltan) > 0) install.packages(faltan)

# Resultado
faltan <- paquetes[!sapply(paquetes, requireNamespace, quietly = TRUE)]
if (length(faltan) == 0) {
  message("Listo. Reinicie R (Ctrl+Shift+F10) y corra el documento.")
} else {
  message("No cargaron: ", paste(faltan, collapse = ", "))
}
