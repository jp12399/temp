import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape == (batch_size, 64, hidden_size)
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # attention_weights shape == (batch_size, 64, 1)
    # we get 1 at the last axis because we are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.CuDNNLSTM(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state, cell = self.lstm(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, cell, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


class AdaptiveAttention(tf.keras.Model):
  def __init__(self, units):
    super(AdaptiveAttention, self).__init__()
    self.S1 = tf.keras.layers.Dense(units)
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, sentinel, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
    concat_feature = tf.concat([features, sentinel], 1)
    concat_feature_emb = tf.concat([self.W1(features), self.S1(sentinel)], 1)
    # hidden shape == (batch_size, 1, hidden_size)
    # score shape == (batch_size, 65, hidden_size)
    score = tf.nn.tanh(concat_feature_emb + self.W2(hidden))

    # attention_weights shape == (batch_size, 65, 1)
    # we get 1 at the last axis because we are applying score to self.V
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * concat_feature
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class CNN_Encoder_Adaptive(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder_Adaptive, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder_Adaptive(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder_Adaptive, self).__init__()
    self.units = units

    self.local_feature_encode = tf.keras.layers.Dense(embedding_dim)
    self.global_feature_encode = tf.keras.layers.Dense(embedding_dim)

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.lstm = tf.keras.layers.CuDNNLSTM(self.units, recurrent_initializer='glorot_uniform', return_sequences=True, return_state=True)

    self.fcg1 = tf.keras.layers.Dense(self.units)
    self.fcg2 = tf.keras.layers.Dense(self.units)

    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = AdaptiveAttention(self.units)

  def call(self, x, img_tensor, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
    # global features shape == (batch_size, 1, embedding_dim)
    features = tf.nn.relu(self.local_feature_encode(img_tensor))
    global_feature = tf.reduce_mean(img_tensor, axis=1, keepdims=True)
    global_feature = tf.nn.relu(self.global_feature_encode(global_feature))
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)
    # inp shape after concat == (batch_size, 1, embedding_dim*2)
    lstm_inp = tf.concat([global_feature, x], axis=2)

    # passing the concatenated vector to the LSTM
    output, state, cell = self.lstm(lstm_inp)

    # sentinel shape == (batch_size, 1, units)
    lstm_inp_re = tf.reshape(lstm_inp, [tf.shape(lstm_inp)[0], tf.shape(lstm_inp)[2]])
    gate = tf.nn.sigmoid(self.fcg1(lstm_inp_re) + self.fcg2(hidden))
    sentinel = tf.expand_dims(tf.multiply(gate, tf.nn.tanh(cell)), 1)

    context, attention_weights = self.attention(features, sentinel, output)

    output = tf.reshape(output, [tf.shape(output)[0], tf.shape(output)[2]])
    mlp_inp = context + output
    # shape == (batch_size, hidden_size)
    x = self.fc1(mlp_inp)

    # output shape == (batch_size, vocab)
    x = self.fc2(x)

    return x, state, cell, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))