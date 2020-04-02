import warnings
warnings. filterwarnings("ignore")


import os
import string
import glob
import cv2


from tensorflow.python.keras.applications.inception_v3 import InceptionV3
import tensorflow.python.keras.applications.inception_v3


#import tensorflow.keras.preprocessing.image
import pickle
#from time import time
import numpy as np
from PIL import Image
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras import Input, layers
from tensorflow.python.keras import optimizers

from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import add
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
import matplotlib.pyplot as plt

START = "startseq"
STOP = "endseq"

root_captioning = "D:\\datasets\\image_caption_data"



encode_model = InceptionV3(weights='imagenet')
encode_model = Model(encode_model.input, encode_model.layers[-2].output)
WIDTH = 299
HEIGHT = 299
OUTPUT_DIM = 2048
preprocess_input = tensorflow.keras.applications.inception_v3.preprocess_input


test_path = os.path.join(root_captioning,"data",f'Vocab.pkl')
with open(test_path, "rb") as fp:
        vocab = pickle.load(fp)


idxtoword = {}
wordtoidx = {}

ix = 1
for w in vocab:
    wordtoidx[w] = ix
    idxtoword[ix] = w
    ix += 1
    
vocab_size = len(idxtoword) + 1 
max_length = 34


test_path = os.path.join(root_captioning,"data",f'embedding_matrix.pkl')
with open(test_path, "rb") as fp:
        embedding_matrix = pickle.load(fp)
        


max_length = 34
embedding_dim = 200
inputs1 = Input(shape=(OUTPUT_DIM,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
caption_model = Model(inputs=[inputs1, inputs2], outputs=outputs)


caption_model.layers[2].set_weights([embedding_matrix])
caption_model.layers[2].trainable = False
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')

model_path = os.path.join(root_captioning,"data",f'caption-model.hdf5')
caption_model.load_weights(model_path)


def getFrameCaptions(video_path):


    def generateCaption(photo):
        in_text = START
        for i in range(max_length):
            sequence = [wordtoidx[w] for w in in_text.split() if w in wordtoidx]
            sequence = pad_sequences([sequence], maxlen=max_length)
            yhat = caption_model.predict([photo,sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = idxtoword[yhat]
            in_text += ' ' + word
            if word == STOP:
                break
        final = in_text.split()
        final = final[1:-1]
        final = ' '.join(final)
        return final

    def encodeImage(img):
    # Resize all images to a standard size (specified bythe image encoding network)
        img = img.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
  # Convert a PIL image to a numpy array
        x = tensorflow.keras.preprocessing.image.img_to_array(img)
  # Expand to 2D array
        x = np.expand_dims(x, axis=0)
  # Perform any preprocessing needed by InceptionV3 or others
        x = preprocess_input(x)
  # Call InceptionV3 (or other) to extract the smaller feature set for the image.
        x = encode_model.predict(x) # Get the encoding vector for the image
  # Shape to correct form to be accepted by LSTM captioning network.
        x = np.reshape(x, OUTPUT_DIM )
        return x

    #image_path = video_path
    cap = cv2.VideoCapture(video_path)

    caption_count = {}
    caption_list = []
    try :
        while (cap.isOpened()):
    
            _,frame = cap.read()
            frame = cv2.resize(frame,(HEIGHT,WIDTH))
        #img = tensorflow.keras.preprocessing.image.load_img(frame, target_size=(HEIGHT, WIDTH))
        #print(img)
        #print("\n\n\n\n\n\n")
        #print(type(img))
        
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            pic = encodeImage(img)
            image = pic.reshape((1,OUTPUT_DIM))
        #plt.imshow(img)
        #plt.show()
            ime = generateCaption(image)
            #print("Caption:",ime)
            #print("_____________________________________")

            if ime in caption_count:
                caption_count[ime] += 1
                continue
            else:
                caption_list.append(ime)
                caption_count.update({ime:1})
    except:

        return caption_count,caption_list