# Название: Определитель растений
# Выполнил: Каштанов Михаил Вячеславович
#
#                                                   Описание проблемы:
#   Некоторые люди не знают названия растений и не могут отлечить одно растение от другого. Поэтому, когда они видят растения/им отправляют фотки
#   они не могут понять, что это за растение. И поэтому, я создал такую программу, которая определяет некоторые виды растений.
#
#                                                   Данные:
#   Датасет с 10 видами цветков: https://drive.google.com/drive/folders/1o-ULYX9UMpD5t6IpOeeRTdnTlEdKf4Fo?usp=sharing
#   Обученная модель: https://drive.google.com/file/d/1ghu7OT2DuYhlDaqGX45bwPCd8S5mQSVH/view?usp=sharing
#   Class_indices.pkl: https://drive.google.com/file/d/1eMr3yHy-2mC-C6PdGylKQA8fGikoZ3Gr/view?usp=sharing
#
#                                                   Использование нейросети/развертывание модели:
#   Данную нейросеть можно использовать как на сайте, по типу virustotal.com (загружаешь картинку, а оно тебе выдает, что за растение), так и в
#   приложении на андроид/пк. Тоже самое, загружешь картинку и оно тебе выдает, что это за цветок и подбирает похожие картинки. На телефоне можно
#   фотографировать цветок/загружать изображение и оно так же будет выдавать предсказание.
#
#                                                   Выводы по работе:
#   Я воплотил в код, то, что хотел. Но, в будущем, я бы хотел сделать примерно тоже самое, но взять уже датасет не на 10 растений, а на 100+,
#   чтобы нейросеть распознавала намного больше растений. А вообще, все это можно импортировать в сайт или приложение. Но для этого уже нужно изучать
#   другие языки, помимо python'а. Мне понравилось создавать эту нейросеть. Конечно, есть уже нейросети, которые уже обучены заранее. Но всегда приятнее
#   сделать что-то самому, и чтобы это "что-то" работало как и хотелось.
#
#
#


# Импортирование всех нужных библиотек

import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import pickle

# Предобработка данных

train_data_dir = r"C:\Users\mik86\OneDrive\Рабочий стол\flowers\train"
test_data_dir = r"C:\Users\mik86\OneDrive\Рабочий стол\flowers\test"
img_height = 150
img_width = 150

data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Загрузка данных из папки с обучающей выборкой

train_data = data_generator.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

# Загрузка данных из папки с тестовой выборкой

test_data = data_generator.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Загрузка предобученной модели VGG16

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Заморозка весов предобученной модели

base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(train_data.num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели

model.fit(train_data, epochs=100)

class_indices = train_data.class_indices

# Сохранение словаря class_indices

with open("class_indices.pkl", "wb") as file:
    pickle.dump(class_indices, file)

# Оценка модели на обучающей выборке

train_evaluation = model.evaluate(train_data)
print("Train Loss: {:.4f}".format(train_evaluation[0]))
print("Train Accuracy: {:.2f}%".format(train_evaluation[1] * 100))

# Оценка модели на тестовой выборке

test_evaluation = model.evaluate(test_data)
print("Test Loss: {:.4f}".format(test_evaluation[0]))
print("Test Accuracy: {:.2f}%".format(test_evaluation[1] * 100))

# Создание графического интерфейса

root = tk.Tk()
root.title("Определение растений")
root.geometry("300x150")

# Создание функции выбора файла

def choose_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = tf.keras.preprocessing.image.load_img(file_path, target_size=(img_height, img_width))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image / 255.0
        image = tf.expand_dims(image, 0)
        prediction = model.predict(image)
        predicted_class_index = tf.argmax(prediction, axis=1).numpy()[0]
        predicted_label = list(train_data.class_indices.keys())[predicted_class_index]
        confidence = prediction[0][predicted_class_index] * 100
        result_label.config(text="Растение определено как: {} ({}%)".format(predicted_label, round(confidence, 2)))
    else:
        result_label.config(text="Файл не выбран")


button = tk.Button(root, text="Выбрать файл", command=choose_file)
button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

# Сохранение модели

model.save("flower_model.h5")

root.mainloop()
