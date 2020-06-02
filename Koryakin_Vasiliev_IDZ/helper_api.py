import os, os.path
import urllib.request
import tarfile
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def download(url, dir='./datasets/', name='dogs.tar'):
    if not os.path.isdir(dir + "dogs") and not os.path.isdir(dir):
        os.mkdir(dir)
        os.mkdir(dir + "dogs")
    else:
        print("Delete datasets dir, and try again")
        return
    # Combine the name and the downloads directory to get the local filename
    filename = os.path.join(dir, name)
    # Download the file if it does not exist
    if not os.path.isfile(filename):
        urllib.request.urlretrieve(url, filename)

    my_tar = tarfile.open(filename)
    my_tar.extractall(dir + "dogs")  # specify which folder to extract to
    my_tar.close()


def loadDataset(path='./datasets/dogs/Images/', labels=5):
    X = []
    Y = []
    # params for cropping and resizing
    sum_w = 0
    sum_h = 0
    valid_images = [".jpg", ".png", ".jpeg"]
    label = 0
    # going by dirs
    for dir in os.listdir(path):
        dir_path = path + str(dir) + "/"
        for f in os.listdir(dir_path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            img = Image.open(os.path.join(dir_path, f))
            img.load()
            # processing params
            w, h = img.size
            X.append(img)
            Y.append(label)
            sum_w = sum_w + w
            sum_h = sum_h + h
        label = label + 1
        if label == labels:
            break
    return X, Y, int(sum_w/len(X)), int(sum_h/len(X))


def showHead(A1, A2, num=6):
    cols = 2
    rows = 3
    for j in range(num):
        fig = plt.figure(figsize=(11, 9))
        for i in range(j*rows,(j*rows)+rows):
            fig.add_subplot(rows, cols, ((i - j * rows) * 2) + 1)
            plt.imshow(A1[i])
            fig.add_subplot(rows, cols, ((i - j * rows) * 2) + 2)
            plt.imshow(A2[i])
        plt.show()
        plt.clf()


def standartize(raw_X, mean_w, mean_h):
    X = []
    for img in raw_X:
        w, h = img.size
        if mean_h > h:
            new_x = int(w * mean_h/float(h))
            img = img.resize((new_x, mean_h))
        w, h = img.size
        if mean_w > w:
            new_y = int(h * mean_w/float(w))
            img = img.resize((mean_w, new_y))
        img = randomCrop(img, mean_w, mean_h).resize((int(mean_w/3), int(mean_h/3)))
        X.append(np.asarray(img))
    return X


def randomCrop(img, cropped_w, cropped_h):
    x, y = img.size
    x1 = np.random.randint(x - cropped_w+1)
    y1 = np.random.randint(y - cropped_h+1)
    return img.crop((x1, y1, x1 + cropped_w, y1 + cropped_h))
