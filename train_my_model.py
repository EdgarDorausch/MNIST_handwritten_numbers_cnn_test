from tensorflow import keras
from tensorflow.python.keras import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Dense, AvgPool2D, Flatten, Dropout, MaxPool2D, MaxPooling2D
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFont, ImageDraw


def drop(f, amount, chunksize=4):
    return [f.read(chunksize) for i in range(amount)]


def get_image_from_data(data, idx):
    img = Image.fromarray(data[idx][0], 'P')
    return img


def max_index(l):
    midx = 0
    m = l[0]
    for i in range(len(l)):
        if l[i] > m:
            m = l[i]
            midx = i
    return midx


def load_data(num, uri_data, uri_labels):
    data = np.empty((num, 1, 28, 28), dtype=np.uint8)
    labels = np.empty((num,), dtype=np.uint8)
    with open(uri_data, "rb") as f_data:
        with open(uri_labels, "rb") as f_labels:
            drop(f_labels, 2)
            drop(f_data, 4)

            for n in range(num):
                print(n, end="                     \r")
                label_buffer = f_labels.read(1)
                labels[n] = int.from_bytes(label_buffer, "little")

                data_buffer = f_data.read(28 * 28)
                data[n][0] = np.frombuffer(data_buffer, dtype=np.uint8).reshape((28, 28))
    return data, labels


def prepare_data(_data, _labels):
    labels = keras.utils.to_categorical(_labels, 10)
    data = _data / 255.0
    return data, labels


def train_model(train_data, train_labels, test_data, test_labels, epochs):
    input_shape = (28, 28, 1)

    my_model = Sequential()
    my_model.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
    my_model.add(Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    my_model.add(MaxPooling2D())
    my_model.add(Dropout(0.25))
    my_model.add(Flatten())
    my_model.add(Dropout(0.5))
    my_model.add(Dense(128, activation='relu'))
    my_model.add(Dense(10, activation='softmax'))

    my_model.compile(loss=keras.losses.categorical_crossentropy,
                     optimizer=keras.optimizers.Adadelta(),
                     metrics=['accuracy'])

    my_model.fit(train_data.reshape((60000, 28, 28, 1)), train_labels,
                 batch_size=128,
                 epochs=epochs,
                 verbose=1,
                 validation_data=(test_data.reshape((10000, 28, 28, 1)), test_labels))
    return my_model


class App:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model: Sequential = None

    def train_model(self, epochs):
        train_data, train_labels = prepare_data(
            *load_data(60000, "./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte"))
        test_data, test_labels = prepare_data(
            *load_data(10000, "./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte"))

        model = train_model(train_data, train_labels, test_data, test_labels, epochs)
        if self.model_path:
            model.save(self.model_path)
        self.model = model

    def load_model(self):
        if self.model_path:
            self.model = keras.models.load_model(self.model_path)
        else:
            print("[WARN]: No model path specified")

    def show_predictions(self, n):
        data, labels = load_data(10000, "./data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte")
        data = data[0:n]
        prep_data = data / 255.0

        probs = self.model.predict(prep_data.reshape((n, 28, 28, 1)))
        preds = [max_index(x) for x in probs[0:n]]

        font = ImageFont.truetype("./Ubuntu-Regular.ttf", 30)
        img_height = 28
        img_width = 28
        scaling = 4
        img_big = Image.new("RGB", (img_width * n * scaling, img_height * scaling))
        for i in range(n):
            img = Image.fromarray(data[i][0], 'P')
            img = img.resize((28 * scaling, 28 * scaling)).convert(mode="RGB")

            d = ImageDraw.Draw(img)
            d.text((0, 0), str(preds[i]), fill=(255, 255, 0), font=font)
            del d

            img_big.paste(img, (i * img_width * scaling, 0))
        img_big.show()


def main():
    app = App("./my_own_model.h5")
    app.train_model(12)

    app.show_predictions(10)


if __name__ == "__main__":
    main()
