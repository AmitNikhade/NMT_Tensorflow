import tensorflow as tf
import numpy as np
import sys
sys.path.append(".")

from Utils import preprocess
import json 
from model import Encoder, Decoder



with open(r'data\config.json', 'r') as openfile: 
    json_object = json.load(openfile)  

max_hin_len = json_object['max_hin_len']
max_eng_len = json_object['max_eng_len']
vocab_eng_size = json_object['vocab_eng_size']
vocab_hin_size = json_object['vocab_hin_size']
embedding_dim = json_object['embedding_dim']
units = json_object['units'] 
bs = json_object['bs']


def evaluate(s,tok_eng, tok_hin):

    encoder = Encoder.Encoder(vocab_eng_size, embedding_dim, units, bs)
    # attention_layer = Attention.BahdanauAttention(10)
    decoder = Decoder.Decoder(vocab_hin_size, embedding_dim, units, bs)

    attention_plot = np.zeros((max_hin_len, max_eng_len))

    sentence = preprocess.preprocess_eng(s)

    inputs = [tok_eng.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_eng_len,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tok_hin.word_index['<s>']], 0)

    for t in range(max_hin_len):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += tok_hin.index_word[predicted_id] + ' '

        if tok_hin.index_word[predicted_id] == '<e>':
            return result, sentence, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot
