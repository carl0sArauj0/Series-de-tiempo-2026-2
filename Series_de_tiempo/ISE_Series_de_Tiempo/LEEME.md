# Análisis descriptivo del ISE (2005-2026)

## Contenido

| Archivo | Para qué sirve |
|---|---|
| `ISE_Series_de_Tiempo.Rproj` | Abre el proyecto. **Siempre empezar por aquí.** |
| `instalar_paquetes.R` | Instala los paquetes que falten. Una sola vez por computador. |
| `Analisis_ISE.qmd` | Documento completo: código, explicaciones e interpretaciones. |
| `Analisis_ISE.R` | El mismo código en formato script, para correr línea por línea. |
| `Analisis_ISE.html` | Resultado ya generado (se puede abrir en el navegador). |
| `datos/ISE2005_2026.xlsx` | La serie del ISE. |

## Primera vez en un computador

1. Descomprimir la carpeta completa (no abrir los archivos desde dentro del .zip).
2. Doble clic en `ISE_Series_de_Tiempo.Rproj`. RStudio abre con la carpeta del proyecto como directorio de trabajo.
3. Abrir `instalar_paquetes.R` y hacer clic en **Source**. Tarda varios minutos la primera vez.

## Uso normal

- **Para ver o modificar el análisis:** abrir `Analisis_ISE.qmd`. Cada bloque gris es un *chunk* de código: se corre con el triángulo verde de su esquina (o Ctrl+Shift+Enter), y una línea suelta con Ctrl+Enter.
- **Para generar el informe:** botón **Render** en `Analisis_ISE.qmd`. Produce `Analisis_ISE.html`.
- **Para trabajar como script:** abrir `Analisis_ISE.R`. El índice de secciones está en el botón **Outline** del editor.

Los datos se leen con `here::here("datos", "ISE2005_2026.xlsx")`, que arma la ruta desde la carpeta del proyecto. No hay que cambiar rutas en ningún computador, siempre que el proyecto se abra desde el `.Rproj`.

Si se modifica el `.qmd`, el `.R` no se actualiza solo: hay que copiar el cambio o trabajar solo en uno de los dos.
