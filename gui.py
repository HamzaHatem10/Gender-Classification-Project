import os
import random
import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import tensorflow as tf

IMG_SIZE = 64
DATASET_PATH = "dataset/UTKFace"

# Load model
model = tf.keras.models.load_model("model/gender_model.h5")

# Get all images
images_list = os.listdir(DATASET_PATH)

current_image = None
current_path = None


def load_random_image():
    global current_image, current_path

    file = random.choice(images_list)
    current_path = os.path.join(DATASET_PATH, file)

    img = cv2.imread(current_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    display_img = cv2.resize(img, (250, 250))

    current_image = img

    img = Image.fromarray(display_img)
    img = ImageTk.PhotoImage(img)

    image_label.config(image=img)
    image_label.image = img
    result_label.config(text="Click Predict")


def predict_gender():
    global current_image

    img = cv2.resize(current_image, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        result = "Female"
    else:
        result = "Male"

    result_label.config(text=f"Prediction: {result}")


# GUI setup
root = tk.Tk()
root.title("Gender Classification GUI")
root.geometry("400x500")

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)

Button(root, text="Next Random Image", command=load_random_image).pack(pady=5)
Button(root, text="Predict Gender", command=predict_gender).pack(pady=5)

# Start with first image
load_random_image()

root.mainloop()