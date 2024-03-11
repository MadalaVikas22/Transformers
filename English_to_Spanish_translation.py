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
# Your format_dataset function prepares the data by vectorizing the input and target sentences and returning them in the
# format required for training. Your make_dataset function creates a TensorFlow dataset from pairs of English and Spanish sentences,
# batches the dataset, and maps the format_dataset function to each batch.

def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return(
        {"encoder_inputs" : eng,
        "decoder_inputs" : spa[:, : -1],
         },
        spa[:, : -1]
    )

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape : {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape : {inputs["decoder_inputs"].shape}')
    print(f'targets.shape : {targets.shape}')

#____________Building the Model____________
# import keras.ops as ops
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        sled.attention = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim= embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation='relu'),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask = None ):
        if mask is not None :
            padding_mask = tf.cast(mask[:, None, :], dtype = "int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query = inputs , value = inputs, key = inputs, attention_mask = padding_mask
        )
        proj_input = self.layernorm_1(input + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    def get_config(self):
        config = super().get_config()
        config.update(
            dict(embed_dim=self.embed_dim, dense_dim=self.dense_dim, num_heads=self.num_heads)
        )
        return config
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embedding = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.cast.shape(inputs)[-1]
        positions= tf.cast.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embedding(positions)
        return embedded_tokens + embedded_positions
    def compute_mask(self, inputs , mask = None):
        if mask is None:
            return None
        else:
            return tf.cast.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

