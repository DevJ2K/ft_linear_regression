# Ft Linear Regression

<img src="./gitimages/miniature.png" alt="Project Overview" width="100%">

## 📄 Overview

The goal of this project is to introduce basic machine learning concepts. I created a program to predict car prices using a simple linear regression model trained with a gradient descent algorithm. The project focuses on a specific example—predicting car price based on mileage—but once complete, you can apply the model to any similar dataset.

You’ll build two programs. The first will predict car prices: it prompts you for a mileage value, then returns the estimated price for that mileage. The second program trains the model by reading a dataset and applying linear regression to adjust the model for accurate predictions.

## 📦 Installation
![Python version](https://img.shields.io/badge/Python-3.10%2B-blue)

Ensure you have Python 3.10 or newer. Download the latest version of Python from the official [Python website](https://www.python.org/downloads/).

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/DevJ2K/ft_linear_regression.git
    cd ft_linear_regression
    ```

2. **(Optional) Create a Virtual Environment:**

    It’s recommended to use a virtual environment for managing dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies:**

    If dependencies are listed in `requirements.txt`, install them with:

    ```bash
    pip install -r requirements.txt
    ```

## 🔍 Usage

1. **Train the Model:**

    Run the training script:

    ```bash
    python3 train_model.py -h
    ```

    This should display:

    ```bash
    usage: train_model.py [-h] -f FILE [-i ITERATIONS] [-lr LEARNING_RATE] [-a]

    Train a model using the provided configuration.

    options:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  Configuration file for training the model.
      -i ITERATIONS, --iterations ITERATIONS
                            Number of iterations for model training (default: 1000).
      -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                            Learning rate influencing model convergence speed (default: 0.01).
      -a, --animate         Enable training curve animation.
    ```

2. **Predict Using the Model:**

    To make predictions based on mileage:

    ```bash
    python3 predict.py -h
    ```

    Expected output:

    ```bash
    usage: predict.py [-h] -f FILE -m MILEAGE [-s] [-g]

    Use a pre-trained model to make predictions based on input values.

    options:
      -h, --help            show this help message and exit
      -f FILE, --file FILE  File containing the trained model's parameters.
      -m MILEAGE, --mileage MILEAGE
                            Mileage for price prediction.
      -s, --steps           Display detailed prediction steps.
      -g, --graph           Visualize prediction results with a graph.
    ```

## 🚀 Examples

### Training the Model

```bash
python3 train_model.py -f data.json -i 1500 -lr 0.01
```

Expected output:

```bash
SUCCESS: Training data saved in '/$REPOSITORY/ft_linear_regression/models/model_data.json'.

** STATISTICS ***********************************
WARNING: Theta trained on standardized data.
Thetaθ(0) (Weight): [-0.85613918]
Thetaθ(1) (Bias): [2.38235357e-16]
Thetaθ - Matrix form:
[[-8.56139178e-01]
    [ 2.38235357e-16]]

** LEARNING INFORMATIONS ************************
Iterations : 1500
Learning Rate : 0.01
Coefficient of determination : 0.7329747078314375

** STANDARDIZATION INFORMATIONS *****************
Mean X (μx): 101066.25
Mean Y (μy): 6331.833333333333
Standard Deviation X (σx): 51565.1899106445
Standard Deviation Y (σy): 1291.8688873961714
```

Example graph output:

<img src="./gitimages/graph_training_output.png" alt="Graph Training Output" width="100%">

### Using the Model for Prediction

```bash
python3 predict.py -f model_data.json -m 170000 -s -g
```

Expected output:

```bash
** STATISTICS ***********************************
WARNING: Theta trained on standardized data.
Thetaθ(0) (Weight): [-0.85613918]
Thetaθ(1) (Bias): [2.38235357e-16]
Thetaθ - Matrix form:
[[-8.56139178e-01]
    [ 2.38235357e-16]]

** LEARNING INFORMATIONS ************************
Iterations : 1500
Learning Rate : 0.01
Coefficient of determination : 0.7329747078314375

** STANDARDIZATION INFORMATIONS *****************
Mean X (μx): 101066.25
Mean Y (μy): 6331.833333333333
Standard Deviation X (σx): 51565.1899106445
Standard Deviation Y (σy): 1291.8688873961714

** 1. STANDARDIZED INPUT ************************
- Standardize input: standardize(x) = (x - μx)/σx
- standardize(170000) = 1.336827230142134

** 2. MATRIX TRANSFORMATION OF INPUT *************
- Transform standardized input to matrix format.
- 1.336827230142134 → [[1.33682723 1.        ]]

** 3. APPLY MODEL ON MATRIX *********************
- Use theta (θ) for matrix product and provide prediction.
- model(X) = X.θ

- model([[1.33682723 1.        ]]) = -1.1445101658636485

** 4. DESTANDARDIZED PREDICTION *****************
- Destandardize for interpretation.
- destandardize(y) = y * μy + σy
- -1.1445101658636485 → 4853.276258745454

PREDICTION → Estimated price for mileage of 170000km: 4853.28€.
```

Example graph output:

<img src="./gitimages/graph_predict_output.png" alt="Graph Predict Output" width="100%">


