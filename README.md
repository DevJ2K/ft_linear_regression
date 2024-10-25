# Ft Linear Regression

<img src="/gitimages/miniature.png" alt="Project Overview" width="100%">

## Overview

![Python version](https://img.shields.io/badge/Python-3.10%2B-blue)

<!-- ![License](https://img.shields.io/badge/License-MIT-green) -->

## Description

This project is focused on creating a program that train and use a linear regression model.

Computorv1 is a project aimed at building a simple equation-solving program. The program will focus on solving polynomial equations of the second degree or lower, using only exponents. The goal is to display the solutions of these equations clearly.

The program will handle the following tasks:

- Display the equation in its reduced form.
- Identify the degree of the equation.
- Show the solution(s) and, if applicable, the nature of the discriminant (positive, negative, or zero).

This project is a part of a series designed to refresh and strengthen mathematical skills, which will be useful for many future projects.

## :package: Installation

Ensure you have Python 3.10 or newer installed. You can download the latest version of Python from the official [Python website](https://www.python.org/downloads/).

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/DevJ2K/ft_linear_regression.git
    cd ft_linear_regression
    ```

2. **(Optional) Create a Virtual Environment:**

    It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    If there are any dependencies, you should list them in a `requirements.txt` file. Install them using:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. First you need to train the model.

```bash
python3 train_model.py -h
```

Should display:
```bash
usage: train_model.py [-h] -f FILE [-i ITERATIONS] [-lr LEARNING_RATE] [-a]

Train a model using the provided configuration.

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  The configuration file for training the model.
  -i ITERATIONS, --iterations ITERATIONS
                        The number of iterations the AI will go through to train itself (default: 1000).
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        The learning rate that influences how quickly the model converges (default: 0.01).
  -a, --animate         Enable animation of the training curves.
```
2. You can predict the price using the mileage by using the program:

```bash
python3 predict.py -h
```
Should display :
```bash
usage: predict.py [-h] -f FILE -m MILEAGE [-s] [-g]

Use a pre-trained model to make predictions based on the provided input value.

options:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  The name of the model file that contains the trained model's parameters.
  -m MILEAGE, --mileage MILEAGE
                        The mileage for which you want to predict the price.
  -s, --steps           If enabled, displays the detailed steps taken during the prediction process.
  -g, --graph           If enabled, visualizes the prediction results with a graph.
```

## Examples

### Train model

```bash
python3 train_model.py -f data.json -i 1500 -lr 0.01
```

Should display :
```bash
SUCCESS: The training has been successfully saved in the file '/home/tajavon/myGithub42/ft_linear_regression/models/model_data.json'.

** STATISTICS ***********************************
WARNING: Theta was trained on standardized data.
Thetaθ(0) (Weight): [-0.85613918]
Thetaθ(1) (Bias): [2.38235357e-16]
Thetaθ - Matrix form:
[[-8.56139178e-01]
 [ 2.38235357e-16]]

** LEARNING INFORMATIONS ************************
Iterations : 1500
Learning Rate : 0.01

** STANDARDIZATION INFORMATIONS *****************
Mean X (μx): 101066.25
Mean Y (μy): 6331.833333333333
Standard Deviation X (σx): 51565.1899106445
Standard Deviation Y (σy): 1291.8688873961714
```

and a graph like this:

<img src="/gitimages/graph_training_output.png" alt="Graph Training Output" width="100%">

### Use model

```bash
python3 predict.py -f model_data.json -m 170000 -s -g
```

Should display :
```bash
** STATISTICS ***********************************
WARNING: Theta was trained on standardized data.
Thetaθ(0) (Weight): [-0.85613918]
Thetaθ(1) (Bias): [2.38235357e-16]
Thetaθ - Matrix form:
[[-8.56139178e-01]
 [ 2.38235357e-16]]

** LEARNING INFORMATIONS ************************
Iterations : 1500
Learning Rate : 0.01

** STANDARDIZATION INFORMATIONS *****************
Mean X (μx): 101066.25
Mean Y (μy): 6331.833333333333
Standard Deviation X (σx): 51565.1899106445
Standard Deviation Y (σy): 1291.8688873961714

** 1. STANDARDIZED INPUT ************************
- The model has been trained on standardized data, so we need to standardize the input before using the model.
- standardize(x) = (x - μx)/σx
- standardize(170000) = 1.336827230142134

** 2. CONVERT STANDARDIZED INPUT TO MATRIX ******
- The model uses matrix product to make predictions, so we need to transform our standardized input into a matrix.
- 1.336827230142134 → [[1.33682723 1.        ]]

** 3. APPLY MODEL ON MATRIX *********************
- The model uses theta (θ) to perform matrix product and provide our prediction.
- model(X) = X.θ

- model([[1.33682723 1.        ]]) = -1.1445101658636485

** 4. DESTANDARDIZED PREDICTION *****************
- Finally, to interpret the model's prediction, we need to destandardize the prediction.
- destandardize(y) = y * μy + σy
- -1.1445101658636485 → 4853.276258745454

PREDICTION → With a mileage of 170000km, the estimate price of your car is 4853.28€.
```

and a graph like this :
<img src="/gitimages/graph_predict_output.png" alt="Graph Predict Output" width="100%">




<!--
- [GitHub Stats Card](#github-stats-card)
    - [Hiding individual stats](#hiding-individual-stats)
    - [Showing additional individual stats](#showing-additional-individual-stats)
    - [Showing icons](#showing-icons)
    - [Themes](#themes)
    - [Customization](#customization) -->

<!-- <img src="/gitimages/make_output.png" width="75%"> -->

<!--
TO-DO

- [x] Curves animation
- [x] Legend on graph
- [ ] Friendly UI to train model and use mileage
- [x] Save useful informations in model.json
- [ ] Clean code
- [ ] ? Create LBC scrapper to get new data
 -->
