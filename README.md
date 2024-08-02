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
  - [Contributing]
  - [License]
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
