import tensorflow as tf
import sys 
sys.path.insert(1, r'C:\Users\amitn\OneDrive\Documents\Projects 2021\NMT')
from model import Attention

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)


    self.attention = Attention.BahdanauAttention(1024)

  def call(self, x, hidden, enc_output):

    context_vector, attention_weights = self.attention(hidden, enc_output)

    x = self.embedding(x)


    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    output, state = self.gru(x)


    output = tf.reshape(output, (-1, output.shape[2]))

    x = self.fc(output)

    return x, state, attention_weights