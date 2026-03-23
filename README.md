# titanic-survival-predictor
# 🚢 Titanic Survival Predictor: Logistic Regression from Scratch

## 📌 Overview
This project predicts the survival probability of passengers on the Titanic based on their Age, Sex, and Passenger Class. 

Unlike most beginner machine learning projects, **this model was built completely from scratch without using any high-level ML libraries like `scikit-learn` or `TensorFlow`.** The core algorithm, gradient descent, cost function, and sigmoid activation were all mathematically implemented using pure Python and NumPy to demonstrate a deep, fundamental understanding of how machine learning engines actually work under the hood.

## 🛠️ Tech Stack
* **Python 3**
* **NumPy:** For matrix math, linear algebra, and gradient descent calculations.
* **Pandas:** For dataset loading, cleaning, and feature engineering.
* **Matplotlib:** For visualizing the L-shaped loss curve and ensuring model convergence.

## 🧠 How It Works
1. **The Math:** The model uses a custom-built Logistic Regression algorithm. It calculates the dot product of passenger features and randomly initialized weights, passes the raw linear score through a custom Sigmoid function, and outputs a probability between 0 and 1.
2. **The Training (`train.py`):** The algorithm runs through thousands of iterations of Gradient Descent, calculating the log-loss (cross-entropy) error and updating the weights using a custom learning rate until the mathematical minimum is found.
3. **The Deployment (`predict.py`):** The final trained weights and biases are saved locally as a `.npz` file. A lightweight prediction script loads this "brain" to instantly classify new user inputs in the terminal.
4. By dividing with the total error by the length of the dataset,the model calculates the mean error insted of the sun of the errors,which prevents the "exploding the gradient problem during training."

## 🚀 Example Output (Jack vs. Rose)
When testing the final model on the historic profiles of the movie characters, the math perfectly aligns with history:
* **Rose (17, Female, 1st Class):** 83.68% Survival Chance
* **Jack (20, Male, 3rd Class):** 15.93% Survival Chance
*
