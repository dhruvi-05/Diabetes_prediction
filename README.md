# Diabetes Prediction

This project uses a machine learning model to predict whether a person has diabetes based on several health parameters. The project includes data preprocessing, training a Support Vector Machine (SVM) classifier, and a Streamlit web application for user interaction.

## Table of Contents

- [Diabetes Prediction]
  - [Table of Contents]
  - [Project Overview]
  - [Data]
  - [Requirements]
  - [Installation]
  - [Usage]
  - [File Structure]
  - [Model Training]
  - [Web Application]
  - [Acknowledgements]

## Project Overview

This project aims to predict diabetes using a Support Vector Machine (SVM) model trained on the Pima Indians Diabetes Database. The model takes several health parameters as input and outputs whether a person is diabetic or not.

## Data

The dataset used in this project is the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database). It contains the following features:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age (years)
- Outcome: Class variable (0 or 1)

## Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- Streamlit
- pickle
- PIL (Pillow)

You can install these requirements using pip:

```bash
pip install numpy pandas scikit-learn streamlit pickle-mixin pillow
```
## Installation

Follow these steps to install the project:
1. Clone the repository:
    ```bash
    git clone https://github.com/my-username/my-project.git
    ```
2. Navigate to the project directory:
    ```bash
    cd my-project
    ```

## Usage

# Model Training

1) Ensure you have the dataset (diabetes.csv) in the project directory.
2) Run the model training script to preprocess data, train the model, and save the trained model and scaler:
   ```bash
   python train_model.py
   ```

## Web Application
1) Run the Streamlit web application:
```bash
streamlit run app.py
```
2) Open your web browser and go to http://localhost:8501 to use the application.


## File Structure
```bash
diabetes-prediction/
│
├── diabetes.csv
├── train_model.py
├── app.py
├── diabetes_classifier.pkl
├── scaler.pkl
└── README.md
```

## Model Training

The model training script (train_model.py) includes the following steps:
1) Load the dataset.
2) Preprocess the data (e.g., handle missing values, scale features).
3) Split the data into training and testing sets.
4) Train a Support Vector Machine (SVM) model.
5) Save the trained model and scaler for later use.

## Web Application

The web application (app.py) uses Streamlit to provide an interactive interface where users can input health parameters and get a prediction on whether they are diabetic or not. The application:
1) Loads the saved model and scaler.
2) Takes user input for the health parameters.
3) Preprocesses the input data.
4) Predicts the outcome using the trained model.
5) Displays the prediction result.

## Acknowledgements

The dataset is sourced from the UCI Machine Learning Repository.
The project is inspired by various machine learning and data science resources.

