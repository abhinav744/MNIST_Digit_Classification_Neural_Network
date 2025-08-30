# ===================================
# MNIST Digit Recognition - Streamlit UI
# ===================================

import numpy as np
import streamlit as st
from tensorflow import keras
from keras.datasets import mnist

# Seed
import tensorflow as tf
tf.random.set_seed(3)

st.title("🧠 MNIST Digit Recognition")
st.write("Train a Neural Network on the MNIST dataset and test predictions interactively.")

# -------------------------------
# Load MNIST data
# -------------------------------
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# -------------------------------
# Build Model
# -------------------------------
model = keras.Sequential([
    keras.layers.Input(shape=(28,28)),
    keras.layers.Flatten(),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

with st.spinner("Training the model... ⏳"):
    model.fit(X_train, Y_train, epochs=5, validation_split=0.1, verbose=0)

# -------------------------------
# Evaluate Model
# -------------------------------
loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
st.success(f"✅ Test Accuracy: {accuracy*100:.2f}%")

# -------------------------------
# Interactive Prediction UI
# -------------------------------
st.subheader("🔍 Test a Prediction")

index = st.slider("Select Image Index", 0, len(X_test)-1, 0)

image = X_test[index]

col1, col2 = st.columns(2)

with col1:
    st.image(image, caption=f"True Label: {Y_test[index]}", width=150)

with col2:
    pred = model.predict(image.reshape(1,28,28), verbose=0)
    pred_label = np.argmax(pred)

    st.write(f"🤖 Predicted Label: **{pred_label}**")
    st.bar_chart(pred[0])
