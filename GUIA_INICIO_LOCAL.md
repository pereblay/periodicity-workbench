# Guía rápida: Periodicity Workbench en local

Esta guía explica cómo instalar Git y Python, descargar el proyecto y arrancar la aplicación en Windows o Linux.

## 1. Instalar Git y Python

### Windows

1. Instala **Git for Windows** desde [git-scm.com/install/win](https://git-scm.com/install/win).
2. Instala Python desde [python.org/downloads](https://www.python.org/downloads/). Python 3.12 o posterior es una buena opción.
3. Abre **PowerShell** y comprueba las instalaciones:

```powershell
git --version
py --version
```

### Linux

Abre una terminal e instala Git, Python, `pip` y el soporte para entornos virtuales. Usa los comandos de tu distribución:

**Ubuntu o Debian:**

```bash
sudo apt update
sudo apt install git python3 python3-venv python3-pip
```

**Fedora:**

```bash
sudo dnf install git python3 python3-pip
```

Comprueba que se instalaron:

```bash
git --version
python3 --version
```

## 2. Descargar el proyecto

En PowerShell o en la terminal de Linux, ve a la carpeta donde quieras guardar el proyecto y ejecuta:

```bash
git clone https://github.com/pereblay/periodicity-workbench.git
cd periodicity-workbench
```

## 3. Crear un entorno e instalar Streamlit y las dependencias

El entorno virtual mantiene las bibliotecas de este proyecto separadas de las del resto del equipo.

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

La instalación puede tardar unos minutos. `requirements.txt` instala Streamlit y las demás bibliotecas que necesita la aplicación.

## 4. Arrancar la aplicación

Con el entorno virtual activado y dentro de la carpeta `periodicity-workbench`, ejecuta:

```bash
python -m streamlit run streamlit_app.py
```

Se abrirá el navegador. Si no se abre automáticamente, visita [http://localhost:8501](http://localhost:8501).

Para detener la aplicación, vuelve a la terminal y pulsa **Ctrl+C**.

## La próxima vez

No hace falta volver a clonar ni instalar las bibliotecas. Entra en la carpeta del proyecto, activa el entorno virtual y arranca la app:

**Windows PowerShell:**

```powershell
cd periodicity-workbench
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

**Linux:**

```bash
cd periodicity-workbench
source .venv/bin/activate
python -m streamlit run streamlit_app.py
```

## Problema habitual en Linux

Si `python3 -m venv .venv` indica que no encuentra `venv`, en Ubuntu o Debian instala el paquete que falta y vuelve a crear el entorno:

```bash
sudo apt install python3-venv
```
