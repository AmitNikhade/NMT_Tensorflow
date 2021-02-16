import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
import re
import pickle

import sys, os
sys.path.append(".")
from Utils import preprocess, Tokenize, train_function, loss
from model import Encoder, Decoder, Attention



from tqdm import tqdm
import string
from sklearn.model_selection import train_test_split
import os
warnings.filterwarnings('ignore')
import json

import argparse

parser = argparse.ArgumentParser(description='Neural Machine Translation (Tensorflow)')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--ns', type=int, default=35000)
args = parser.parse_args()

file1 = open(r"data\config.json","a") 


tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)

PATH = r'data\Hindi_English_Truncated_Corpus.csv'
embedding_dim = 256
epochs = args.epochs
bs = args.batch_size
ns = args.ns
units = 1024
checkpoint_dir = r'src\checkpoints'

df = pd.read_csv(PATH)
df["eng_sent_len"] = df["english_sentence"].apply(lambda x: len(str(x).split(' ')))
df["hindi_sent_len"] = df["hindi_sentence"].apply(lambda x: len(str(x).split(' ')))

df = df.loc[df['hindi_sent_len'] < 30].copy()

df['english_sentence'] = df['english_sentence'].apply(preprocess.preprocess_eng)
df['hindi_sentence'] = df['hindi_sentence'].apply(preprocess.preprocess_hindi)

hindi = df['hindi_sentence'].values.tolist()[:ns]
english = df['english_sentence'].values.tolist()[:ns]
hin, tok_hin = Tokenize.tokenize(hindi)
eng, tok_eng = Tokenize.tokenize(english)

with open(r'data\tok_h.pickle', 'wb') as tok1:
    pickle.dump(tok_hin, tok1, protocol=pickle.HIGHEST_PROTOCOL)
with open(r'data\tok_e.pickle', 'wb') as tok2:
    pickle.dump(tok_eng, tok2, protocol=pickle.HIGHEST_PROTOCOL)

vocab_eng_size = len(tok_eng.word_index)+1
vocab_hin_size = len(tok_hin.word_index)+1

max_hin_len = max(len(t) for t in hin)
max_eng_len = max(len(t) for t in eng)

X_train, X_test, y_train, y_test = train_test_split(eng, hin, test_size=0.2)

steps_per_epoch = len(X_train)//bs

optimizer = tf.keras.optimizers.Adam()

encoder = Encoder.Encoder(vocab_eng_size, embedding_dim, units, bs)
attention_layer = Attention.BahdanauAttention(10)
decoder = Decoder.Decoder(vocab_hin_size, embedding_dim, units, bs)


checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


BUFFER_SIZE = len(X_train)
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(bs, drop_remainder=True)

dictio ={ 
    "max_hin_len" : max_hin_len, 
    "max_eng_len" : max_eng_len,
    "vocab_eng_size": vocab_eng_size,
    "vocab_hin_size": vocab_eng_size,
    "units": units,
    "embedding_dim":embedding_dim,
    "bs": bs
} 
json_obj = json.dumps(dictio) 
print(json_obj)

with open(r"data\config.json", "w") as outfile: 
    outfile.write(json_obj) 


for epoch in range(epochs):

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_function.train_step(inp, targ, enc_hidden, encoder, decoder, tok_hin, optimizer, loss.loss_function, bs)
    total_loss += batch_loss

    if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))

  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))