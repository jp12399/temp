import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import json

def load_data(data_kind='train', num_examples=10000):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                             cache_subdir=os.path.abspath('.'),
                                             origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                             extract=True)
    annotation_file = os.path.dirname(annotation_zip) + '/annotations/captions_' + data_kind + '2014.json'

    name_of_zip = data_kind + '2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/' + data_kind + '2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + '/' + data_kind + '2014/'
    else:
        PATH = os.path.abspath('.') + '/' + data_kind + '2014/'

    # read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # storing the captions and the image name in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_' + data_kind + '2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # shuffling the captions and image_names together
    # setting a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    # selecting the first ~ captions from the shuffled set
    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    return img_name_vector, train_captions


def load_image(image_path):
    print(image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize_images(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def preprocess_caption(captions, top_k=5000):

    # This will find the maximum length of any caption in our dataset
    def calc_max_length(tensor):
        return max(len(t) for t in tensor)

    # The steps above is a general process of dealing with text processing

    # choosing the top ~ words from the vocabulary
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(captions)
    train_seqs = tokenizer.texts_to_sequences(captions)


    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'


    # creating the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(captions)

    # padding each vector to the max_length of the captions
    # if the max_length parameter is not provided, pad_sequences calculates that automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # calculating the max_length
    # used to store the attention weights
    max_length = calc_max_length(train_seqs)
    return cap_vector, tokenizer, max_length


def load_batch(img_name, caption, batch_size=64, buffer_size=1000):
    # loading the numpy files
    def map_func(img_name, cap):
        img_tensor = np.load(img_name.decode('utf-8') + '.npy')
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((img_name, caption))

    # using map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.py_func(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.contrib.data.AUTOTUNE)

    # shuffling and batching
    dataset = dataset.shuffle(buffer_size).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

    return dataset


def save_feature(img_name_vector, image_features_extract_model):
    tf.enable_eager_execution()
    encode_train = sorted(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    for img, path in image_dataset:
      batch_features = image_features_extract_model(img)
      batch_features = tf.reshape(batch_features,
                                  (batch_features.shape[0], -1, batch_features.shape[3]))

      for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())

    return
