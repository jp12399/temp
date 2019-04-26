import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from data import load_data, preprocess_caption, load_batch, save_feature
from model import CNN_Encoder, RNN_Decoder
from eval_tool import loss_function
from train_tool import train_step, val_step


BATCH_SIZE = 64
BUFFER_SIZE = 1000
EPOCHS = 100
embedding_dim = 256
units = 512


tf.enable_eager_execution()

img_name_vector, train_captions = load_data('train', 384000)

# image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
# new_input = image_model.input
# hidden_layer = image_model.layers[-1].output
# image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
# save_feature(img_name_vector, image_features_extract_model)
#
cap_vector, tokenizer, max_length = preprocess_caption(train_captions, 5000)

# Create training and validation sets using 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector, cap_vector, test_size=0.1, random_state=0)



vocab_size = len(tokenizer.word_index) + 1
print('vocab_size:' + str(vocab_size))
num_steps = len(img_name_train) // BATCH_SIZE
val_num_steps = len(img_name_val) // BATCH_SIZE
# shape of the vector extracted from InceptionV3 is (64, 2048)
# these two variables represent that
features_shape = 2048
attention_features_shape = 64

dataset = load_batch(img_name_train, cap_train, BATCH_SIZE, BUFFER_SIZE)
val_dataset = load_batch(img_name_val, cap_val, BATCH_SIZE, BUFFER_SIZE)


encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.train.AdamOptimizer()


checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=50)


start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])


# adding this in a separate cell because if you run the training cell
# many times, the loss_plot array will be reset
loss_plot = []


for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    val_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, tokenizer, optimizer, BATCH_SIZE)
        total_loss += t_loss

        if batch % 1000 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

    for (batch, (img_tensor, target)) in enumerate(val_dataset):
        batch_loss, t_loss = val_step(img_tensor, target, encoder, decoder, tokenizer, BATCH_SIZE)
        val_loss += t_loss
    # storing the epoch end loss value to plot later
    loss_plot.append(total_loss / num_steps)

    ckpt_manager.save()

    print ('Epoch {} Train Loss {:.6f}'.format(epoch + 1, total_loss / num_steps))
    print('Epoch {} Val Loss {:.6f}'.format(epoch + 1, val_loss / val_num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
