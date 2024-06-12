# Dog Segmentation

![License](https://img.shields.io/github/license/AlvaroPorcel/Dog_Segmentation)
![Issues](https://img.shields.io/github/issues/AlvaroPorcel/Dog_Segmentation)
![Stars](https://img.shields.io/github/stars/AlvaroPorcel/Dog_Segmentation)
![Forks](https://img.shields.io/github/forks/AlvaroPorcel/Dog_Segmentation)

## Description

The **Dog Segmentation** project focuses on segmenting images of dogs using advanced deep learning techniques. By utilizing a convolutional neural network architecture, the model can accurately identify and segment areas of images where dogs are present.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

Follow these steps to set up the environment and run the project locally.

1. Clone the repository:
    ```sh
    git clone https://github.com/AlvaroPorcel/Dog_Segmentation.git
    cd Dog_Segmentation
    ```

2. Create a virtual environment:
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the segmentation model, use the following command:

```sh
python scripts/segment.py --input image.jpg --output result.png
