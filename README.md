# 🖋 MNIST Digit Classification – Neural Network

This project implements a neural network to accurately classify handwritten digits using the famous MNIST dataset—a staple benchmark in computer vision and deep learning.

[🔗 Live Demo](https://wkwdzefxbsqvfoymfzdtnu.streamlit.app/)

## 📌 Project Overview

Objective: Develop and train a neural network model to recognize digits (0–9) from the MNIST dataset (28×28 grayscale images).

Model: A feedforward neural network (e.g., Dense layers using TensorFlow/Keras or PyTorch) trained for high accuracy on test data.

Features:

Data loading, normalization, and preprocessing

Model architecture setup and training pipeline

Evaluation using metrics like accuracy and loss tracking

Interactive Streamlit app for live predictions

## 📂 Repository Structure

/MNIST_Digit_Classification_Neural_Network

│── mnist_classification.ipynb     # Notebook with data, model & analysis

│── app.py                         # Streamlit app for live predictions

│── requirements.txt               # Dependencies

│── README.md                      # This file!

## ⚙ Getting Started

### 1. Clone this repository:

git clone https://github.com/abhinav744/MNIST_Digit_Classification_Neural_Network.git

cd MNIST_Digit_Classification_Neural_Network

### 2. (Optional) Create a virtual environment:

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install dependencies:

pip install -r requirements.txt

### 4. Run the Notebook:

jupyter notebook mnist_classification.ipynb

## 📊 Results & Insights

Achieved ~98% accuracy on MNIST test set.

Demonstrates strong performance for a simple dense NN, but can be further improved with CNNs.

## 🔮 Future Enhancements

Explore Convolutional Neural Networks (CNNs) for improved accuracy

Add dropout & regularization to reduce overfitting

Enhance the Streamlit UI with drawing canvas for digit input

Visualize confusion matrix and misclassified samples
