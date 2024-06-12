# Dog_Segmentation


![License](https://img.shields.io/github/license/AlvaroPorcel/Dog_Segmentation)
![Issues](https://img.shields.io/github/issues/AlvaroPorcel/Dog_Segmentation)
![Stars](https://img.shields.io/github/stars/AlvaroPorcel/Dog_Segmentation)
![Forks](https://img.shields.io/github/forks/AlvaroPorcel/Dog_Segmentation)

## Descripción

El proyecto **Dog Segmentation** se centra en la segmentación de imágenes de perros mediante técnicas avanzadas de aprendizaje profundo. Utilizando una arquitectura de red neuronal convolucional, el modelo es capaz de identificar y segmentar con precisión las áreas de las imágenes donde se encuentran los perros.

## Tabla de Contenidos

- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribución](#contribución)
- [Licencia](#licencia)
- [Contacto](#contacto)

## Instalación

Sigue estos pasos para configurar el entorno y ejecutar el proyecto localmente.

1. Clonar el repositorio:
    ```sh
    git clone https://github.com/AlvaroPorcel/Dog_Segmentation.git
    cd Dog_Segmentation
    ```

2. Crear un entorno virtual:
    ```sh
    python -m venv env
    source env/bin/activate  # En Windows usa `env\Scripts\activate`
    ```

3. Instalar las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

## Uso

Para ejecutar el modelo de segmentación, usa el siguiente comando:

```sh
python segment.py --input imagen.jpg --output resultado.png
