# Periodicity Workbench — Desktop GUI

Esta carpeta contiene la nueva GUI de escritorio basada en Qt/PySide6. Las pestañas comparten un mismo estado de trabajo; Lomb–Scargle y el bootstrap se ejecutan en un hilo de cálculo local para que la ventana siga respondiendo. El perfil plegado tiene una pestaña propia, y el *sampling window* aparece debajo del periodograma.

## Primera instalación (macOS/Linux)

Ejecuta una vez:

```bash
./install.sh
```

El instalador elige Python 3.14, 3.13 o 3.12, crea `.venv` e instala ruedas binarias de Qt y de las dependencias científicas. Esta versión ya no instala Streamlit ni PyArrow.

## Arranques siguientes

```bash
./run.sh
```

La ventana de escritorio se abre directamente. No es necesario reinstalar las dependencias en cada arranque.


## Guardar y recuperar sesiones

En la pestaña **Session**, guarda un paquete `.pwb` con la curva original, los resultados calculados, la configuración y las evidencias del alumno. Para continuar en otro momento, abre ese paquete. El contenido se guarda en el ordenador del alumno.

La restauración carga los resultados guardados; no vuelve a ejecutar Lomb–Scargle ni el bootstrap.

## Migración de funciones

La primera conversión nativa incluye carga ASCII/FITS, acción en Frequency Search para restar o restaurar el baseline ponderado del flujo, resolución de objetos con SIMBAD, búsqueda bibliográfica en una ventana emergente, búsqueda Lomb–Scargle, estimación de errores mediante un botón de bootstrap independiente que reutiliza los picos del análisis, gráfico de sampling window calculado una vez por entrada y reutilizado en los análisis siguientes, perfil plegado, evidencias y sesiones guardadas. Los paneles de análisis avanzado y generación del informe PDF todavía están pendientes de migrar desde la versión anterior.

## Criterio visual

La interfaz debe mantener los campos de texto y selectores compactos y proporcionados. Evita controles de una sola línea muy largos en todas las pestañas; reserva el espacio amplio para gráficas y para la edición del informe. En la pestaña Input, ASCII y FITS son alternativas visibles lado a lado, la gráfica de datos solo se actualiza al pulsar **Plot selected columns**, y el zoom sincroniza los límites del análisis. Al volver al encuadre inicial o alejarse más, esos límites regresan a `auto`. El botón **Send to frequency analysis** valida y fija los datos y opciones que usarán las pestañas de análisis.

La paleta toma como referencia la identidad visual de VIU: naranja como acento, superficies blancas y texto oscuro.

Las gráficas incorporan controles tipo Matplotlib para pan, zoom rectangular por arrastre, acercar, alejar y volver a la vista inicial.
