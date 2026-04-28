import os
import cv2
import numpy as np

IMG_SIZE = 64

def load_dataset(path, limit=8000):
    images = []
    labels = []

    files = os.listdir(path)
    
    for i, file in enumerate(files):
        if i >= limit:
            break
        try:
            gender = int(file.split("_")[1])

            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            images.append(img)
            labels.append(gender)

        except:
            continue

    return np.array(images), np.array(labels)