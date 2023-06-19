# Импортирование нужных библиотек

import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import pickle

# Загрузка сохраненной модели

model = load_model("flower_model.h5")

# Загрузка словаря class_indices

with open("class_indices.pkl", "rb") as file:
    class_indices = pickle.load(file)

# Размеры изображения

img_height = 150
img_width = 150

# Создание графического интерфейса

root = tk.Tk()
root.title("Определение растений")
root.geometry("300x150")

# Создание функции выбора файла

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = load_img(file_path, target_size=(img_height, img_width))
        image = img_to_array(image) / 255.0
        image = tf.expand_dims(image, 0)
        prediction = model.predict(image)
        predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
        predicted_label = list(class_indices.keys())[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100
        result_label.config(text="Растение определено как: {} ({}%)".format(predicted_label, round(confidence, 2)))
    else:
        result_label.config(text="Файл не выбран")


button = tk.Button(root, text="Выбрать файл", command=choose_file)
button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
