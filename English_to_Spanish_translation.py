import os
os.environ['KERAS_BACKEND'] = "tensorflow"

import pathlib
import random
import re
import string
import numpy as np

import keras
import tensorflow as tf
import tensorflow.data as tf_data
from tensorflow import strings as tf_strings
from keras import layers
#from keras import ops as ops
from keras.layers import TextVectorization

#_______________Downloading the Data_______________

text_file = keras.utils.get_file(
    fname= "spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"
#_______________Parsing the Data_______________

with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []
for line in lines:
    parts = line.split("\t")
    if len(parts) == 2:
        eng, spa = parts
        spa = "[start] " + spa + " [end]"
        text_pairs.append((eng, spa))
    #else:
        #print(f"Ignoring line with unexpected format: {line}")


for i in range(5):
    print((random.choice(text_pairs)))

#_______________Splitting the Data_______________
random.shuffle(text_pairs)
num_validation_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_validation_samples

train_pairs = text_pairs[: num_train_samples]
val_pairs = text_pairs[num_train_samples: num_train_samples + num_validation_samples]
test_pairs = text_pairs[num_train_samples+num_validation_samples : ]

print(f"Total pairs : {len(text_pairs)}" )
print(f"Train pairs : {len(train_pairs)}" )
print(f"Validation pairs : {len(val_pairs)}" )
print(f"Test pairs : {len(test_pairs)}" )

#_______________Vectorizing the Text Data_______________

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

def custom_standardization(input_string):
    lowercase = tf_strings.lower(input_string)
    return tf_strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

eng_vectorization = TextVectorization(
    max_tokens = vocab_size,
    output_mode = "int",
    output_sequence_length= sequence_length,
)

spa_vectorization = TextVectorization(
    max_tokens= vocab_size,
    output_mode= "int",
    output_sequence_length= sequence_length + 1,
    standardize= custom_standardization,
)

train_eng_texts = [pairs[0] for pairs in train_pairs]
train_spa_texts = [pairs[1] for pairs in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)

