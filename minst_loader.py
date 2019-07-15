from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(y_train, y_train.shape)
# input image dimensions
img_rows, img_cols = 28, 28


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)





def max_index(l):
  midx = 0
  m = l[0]
  for i in range(len(l)):
    if l[i] > m:
      m = l[i]
      midx = i
  return midx

# Recreate the exact same model purely from the file
n = 20

new_model = keras.models.load_model('./my_model.h5')
probs = new_model.predict(x_test)
preds = [max_index(x) for x in probs[0:n]]

img_height = 28
img_width = 28
scaling = 4

font = ImageFont.truetype("/home/edgar/.fonts/f/FiraSans_Regular.ttf", 30)

img_big = Image.new("RGB", (img_width * n * scaling, img_height * scaling))
for i in range(n):
  img = Image.fromarray(x_test.reshape((1,10000,28,28))[0][i], 'P')
  img = img.resize((28*scaling, 28*scaling)).convert(mode="RGB")

  d = ImageDraw.Draw(img)
  d.text((0,0), str(preds[i]), fill=(255,255,0), font=font)
  del d

  img_big.paste(img, (i*img_width*scaling,0))
# img_big = img_big.resize((img_big.size[0]*scaling, img_big.size[1]*scaling))

img_big.show()
print(y_train)