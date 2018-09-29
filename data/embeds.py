import os
import numpy as np
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input
import PIL
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
import tensorflow as tf



def create_inception_embedding(filenames,inception):
    grayscaled_rgb_resized = []
    for name in filenames:
        i=np.array(PIL.Image.open(name))
        i=gray2rgb(rgb2gray(i))
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

def main():
    train_dir = os.path.join('Dataset', 'Train')
    test_dir = os.path.join('Dataset', 'Test')

    # Get the file names from the train and dev sets
    train_filenames = np.array([os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.jpg')])
    test_filenames  = np.array([os.path.join(test_dir, f)  for f in os.listdir(test_dir)  if f.endswith('.jpg')])

    print('Loading weights for Inception...')
    inception = InceptionResNetV2(weights=None, include_top=True)
    inception.load_weights('inception_resnet_v2_weights_tf_dim_ordering_tf_kernels.h5')
    inception.graph = tf.get_default_graph()
    print('Weights loaded')

    #calculate embeddings
    train_embeds = create_inception_embedding(train_filenames, inception)
    test_embeds = create_inception_embedding(test_filenames, inception)
    np.save('train_embeds',train_embeds)
    np.save('test_embeds', test_embeds)


if __name__ == '__main__':
    try:
        main()

    except Exception as e:
        print('Missing or invalid arguments %s' % e)