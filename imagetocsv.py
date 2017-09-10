import os
import numpy as np
import re
from PIL import Image
import copy
import csv

TRAIN_DIR ='C:\\cf10\\train\\train\\'
train_img = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
# print(len(train_img))
ordered_files = sorted(train_img, key=lambda x: (int(re.sub('\D','',x)),x))

# im = Image.open(ordered_files[1])
# im2 = np.array(im).flatten()
# print(len(im2)) #3072. 32x32 픽셀 x 3가지값(rgb)

data = np.zeros((50000, 3072), dtype=np.float32)
print(data.shape)

for i in range(len(ordered_files)):
    im = Image.open(ordered_files[i])
    im2 = np.array(im).flatten()
    data[i] = im2

def scale(x):
    return (x-np.mean(x))/ np.std(x)

data2 = copy.deepcopy(data)

for i in range(3072):
    data2[i, :] = scale(data2[i,:])


f = open('cf_test.csv', 'w', newline='')
wr = csv.writer(f)
for i in range(50000):
    wr.writerow(data2[i])
f.close()