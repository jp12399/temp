import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from data import load_image


def evaluate(image, encoder, decoder, tokenizer, features_extract_model, beam_search=False, beam_size=3, max_length=18, attention_features_shape=64):

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    if beam_search == True:
        result_num = np.zeros([beam_size, max_length], int)
        next_result_num = np.zeros([beam_size, max_length], int)
        dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
        prev_input = tf.tile(dec_input, [beam_size,1]).numpy() # [beam_size, 1]
        img_input = tf.tile(features, [beam_size,1,1]) # [beam_size, size, nfilters]
        pre_selected_score = np.ones(beam_size, float)
        attention_plot = np.zeros([beam_size, max_length, attention_features_shape], float)
        next_attention_plot = np.zeros([beam_size, max_length, attention_features_shape], float)
        hidden = decoder.reset_state(batch_size=beam_size).numpy()

        for seq_cnt in range(max_length):
            predictions_t, hidden_out_t, cell, attention_weights_t = decoder(
                tf.convert_to_tensor(prev_input), img_input, tf.convert_to_tensor(hidden))
            predictions = tf.nn.softmax(predictions_t).numpy()
            hidden_out = hidden_out_t.numpy()
            attention_weights = np.reshape(attention_weights_t.numpy(), [beam_size, -1])
            selected_idx = np.zeros([beam_size, 1], int)
            score_selected = np.zeros(beam_size, float)
            for beam_cnt in range(beam_size):
                end_exist = False
                for i in range(seq_cnt):
                    if tokenizer.index_word[result_num[i]] == '<end>':
                        end_exist = True
                        break
                if seq_cnt == 0 and beam_cnt > 0:
                    predictions[beam_cnt] = 0
                elif end_exist:
                    predictions[beam_cnt] = 0
                    predictions[beam_cnt, 0] = 1
                else:
                    predictions[beam_cnt] *= pre_selected_score[beam_cnt]
            for beam_cnt in range(beam_size):
                maxidx = np.unravel_index(np.argmax(predictions, axis=None), predictions.shape)
                score_selected[beam_cnt] = predictions[maxidx]
                selected_idx[beam_cnt, 0] = maxidx[1]
                predictions[maxidx] = 0
                if tokenizer.index_word[maxidx[1]] == '<end>':# if eof, turn into zombie
                    predictions[maxidx[0]] = 0
                hidden[beam_cnt] = hidden_out[0][maxidx[0]]
                next_attention_plot[beam_cnt, :seq_cnt] = attention_plot[maxidx[0], :seq_cnt]
                next_attention_plot[beam_cnt, seq_cnt] = attention_weights[maxidx[0]]
                next_result_num[beam_cnt, :seq_cnt] = result_num[maxidx[0], :seq_cnt]
                next_result_num[beam_cnt, seq_cnt] = maxidx[1]
                #print(next_result_num[beam_cnt, :seq_cnt+1])
            prev_input[:] = selected_idx[:]
            pre_selected_score[:] = score_selected[:]
            result_num[:] = next_result_num[:]
            attention_plot[:] = next_attention_plot[:]
        for i in range(beam_size):
            one_line=[]
            for j in range(max_length):
                one_line.append(tokenizer.index_word[result_num[i, j]])
            print(one_line)
        for i in range(max_length):
            result_word = tokenizer.index_word[result_num[0, i]]
            result.append(result_word)
            if result_word == '<end>':
                break
        attention_plot = attention_plot[0]
    else:
        for i in range(max_length):
            predictions, hidden, cell, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            print(predictions[0, 4].numpy())
            print(cell[0,:5].numpy())

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(tokenizer.index_word[predicted_id])

            if tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot

            dec_input = tf.expand_dims([predicted_id], 0)

        attention_plot = attention_plot[:len(result), :]

    return result, attention_plot


def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l,1:], (8, 8))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = tf.keras.backend.sparse_categorical_crossentropy(real, pred, from_logits=True)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)
