import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
from PIL import Image
import pickle
import data
from data import load_data, load_image, preprocess_caption, load_batch
from model import CNN_Encoder_Adaptive, RNN_Decoder_Adaptive
from eval_tool import evaluate, plot_attention

tf.enable_eager_execution()

img_name_vector, train_captions = load_data('train', 320000)
val_img_name_vector, val_captions = load_data('val', 30000)

image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# The steps above is a general process of dealing with text processing
cap_vector, tokenizer, max_length = preprocess_caption(train_captions, 5000)

val_seqs = tokenizer.texts_to_sequences(val_captions)
val_cap_vector = tf.keras.preprocessing.sequence.pad_sequences(val_seqs, padding='post')

# Create training and validation sets using 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

# feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 512
units = 512
vocab_size = len(tokenizer.word_index) + 1
num_steps = len(img_name_train) // BATCH_SIZE
# shape of the vector extracted from InceptionV3 is (64, 2048)
# these two variables represent that
features_shape = 2048
attention_features_shape = 65


batch = load_batch(img_name_train, cap_train)

encoder = CNN_Encoder_Adaptive(embedding_dim)
decoder = RNN_Decoder_Adaptive(embedding_dim, units, vocab_size)

checkpoint_path = "./checkpoints/train_adaptive"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder)
status = ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))


# captions on the validation set
for i in range(len(val_img_name_vector)):
    rid = np.random.randint(0, len(val_img_name_vector))
    image = val_img_name_vector[rid]
    real_caption = ' '.join([tokenizer.index_word[i] for i in val_cap_vector[rid] if i not in [0]])
    result, attention_plot = evaluate(image, encoder, decoder, tokenizer, image_features_extract_model, max_length=max_length, attention_features_shape=attention_features_shape)

    print ('Real Caption:', real_caption)
    print ('Prediction Caption:', ' '.join(result))
    plot_attention(image, result, attention_plot)
    # opening the image
    Image.open(val_img_name_vector[rid])
