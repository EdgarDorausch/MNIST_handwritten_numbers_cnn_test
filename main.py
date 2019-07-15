from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image


def read_labels():
  with open("./data/train-labels-idx1-ubyte", "rb") as labels:
    for i in range(2):
      chunk = f.read(4)
      print(chunk.hex())
    for i in range(10):
      chunk = f.read(1)
      print(int.from_bytes(chunk,"big"), chunk.hex())

def drop(f, amount, chunksize=4):
  return [f.read(chunksize) for i in range(amount)]

def show_numbers():
  with open("./data/train-images-idx3-ubyte", "rb") as data:
    with open("./data/train-labels-idx1-ubyte", "rb") as labels:
      drop(labels, 2)
      drop(data, 4)

      img_height = 28
      img_width = 28

      n = 100
      img_big = Image.new("P", (img_width * 10, img_height * n))
      numbers = [[] for i in range(10)]
      # Read image data
      while(any([len(l) < n for l in numbers])):
        idx = int.from_bytes(labels.read(1), "big")
        if len(numbers[idx]) < n:
          chunk = data.read(img_height*img_width)
          img = Image.frombytes("P",(img_width,img_height),chunk)
          numbers[idx].append(img)
        else:
          drop(data,img_width, img_height)
          
      for x in range(10):
        for y in range(n):
          img_big.paste(numbers[x][y], (x*img_width,y*img_height))

      scaling = 4
      img_big = img_big.resize((img_big.size[0]*scaling, img_big.size[1]*scaling))

      img_big.show()

if __name__ == "__main__":
  show_numbers()