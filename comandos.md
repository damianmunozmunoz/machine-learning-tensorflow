Crear un entorno de trabajo para evitar incompatibilidades en dependencias.
    - Abrimos el anaconda prompt y nos posicionamos en la carpeta que queramos usar
    conda create -n nombre-del-entorno
    - Con Visual Studio Code
    python -m venv nombre-del-entorno

Activar el entorno de trabajo
    - También en el anaconda prompt
    conda activate nombre-del-entorno
    - Con Visual Studio Code
    nombre-del-entorno\Scripts\activate

Instalar pip
    - Con anaconda
    conda install pip
    - Con Visual Studio code
    .\.nombre-del-entorno\Scripts\python.exe -m pip install --upgrade pip

Listar las dependencias instaladas
    pip list

Instalar dependencias
    - openCV
    pip install opencv-contrib-python
    - matplotlib
    pip install matplotlib
    - guiqwt
    pip install guiqwt
    