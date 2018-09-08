import os
import numpy as np
import PIL
from skimage.transform import resize
from skimage.io import imsave


def main():
    train_dir = os.path.join('Dataset', 'Train')
    test_dir = os.path.join('Dataset', 'Test')

        # Get the file names from the train and dev sets
    train_filenames = np.array([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')])
    test_filenames  = np.array([os.path.join(test_dir, f)  for f in os.listdir(test_dir)  if f.endswith('.jpg')])

    for name in train_filenames:
        i = np.array(PIL.Image.open(name))
        i = resize(i, (256, 256, 3), mode='constant')
        imsave(name,i)
    for name in test_filenames:
        i = np.array(PIL.Image.open(name))
        i = resize(i, (256, 256, 3), mode='constant')
        imsave(name,i)



if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        print('Missing or invalid arguments %s' % e)