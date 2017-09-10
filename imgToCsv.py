import numpy as np
import csv

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


#####################################################################################################
label = np.loadtxt('C:\\cf10\\trainLabels.csv', delimiter=',', usecols=[1], dtype=str,skiprows=1)
# print(label) # "b'truck'"
test_label = np.zeros((50000,10), dtype=np.int)
for i in range(len(label)):
    if label[i] == "b'airplane'":   test_label[i,]=[1,0,0,0,0,0,0,0,0,0]
    elif label[i] == "b'automobile'":   test_label[i,]=[0,1,0,0,0,0,0,0,0,0]
    elif label[i] == "b'bird'":   test_label[i,] = [0,0,1,0,0,0,0,0,0,0]
    elif label[i] == "b'cat'":   test_label[i,]=[0,0,0,1,0,0,0,0,0,0]
    elif label[i] == "b'deer'":   test_label[i,] = [0,0,0,0,1,0,0,0,0,0]
    elif label[i] == "b'dog'":   test_label[i,]=[0,0,0,0,0,1,0,0,0,0]
    elif label[i] == "b'frog'":   test_label[i,] = [0,0,0,0,0,0,1,0,0,0]
    elif label[i] == "b'hose'":   test_label[i,]=[0,0,0,0,0,0,0,1,0,0]
    elif label[i] == "b'ship'":   test_label[i,] = [0,0,0,0,0,0,0,0,1,0]
    else:   test_label[i,]=[0,0,0,0,0,0,0,0,0,1]

print(test_label[9])

f = open('test_label.csv', 'w', newline='')
wr = csv.writer(f)
for i in range(50000):
    wr.writerow(test_label[i])
f.close()


