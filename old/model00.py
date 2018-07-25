#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 01:53:28 2018

@author: venkatraman
"""

from glob import glob
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm

import tensorflow as tf

def parser(record, training=True):
    """
    In training mode labels will be returned, otherwise they won't be
    """
    keys_to_features = {
        "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
        "mean_audio": tf.FixedLenFeature([128], tf.float32)
    }
    
    if training:
        keys_to_features["labels"] =  tf.VarLenFeature(tf.int64)
    
    parsed = tf.parse_single_example(record, keys_to_features)
    x = tf.concat([parsed["mean_rgb"], parsed["mean_audio"]], axis=0)
    if training:
        y = tf.sparse_to_dense(parsed["labels"].values, [3862], 1)
        return x, y
    else:
        x = tf.concat([parsed["mean_rgb"], parsed["mean_audio"]], axis=0)
        return x
     
def make_datasetprovider(tf_records, repeats=1000, num_parallel_calls=12, 
                         batch_size=32): 
    """
    tf_records: list of strings - tf records you are going to use.
    repeats: how many times you want to iterate over the data.
    """
    dataset = tf.data.TFRecordDataset(tf_records)
    dataset = dataset.map(map_func=parser)
    dataset = dataset.repeat(repeats)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)

    d_iter = dataset.make_one_shot_iterator()
    return d_iter

def data_generator(tf_records, batch_size=1, repeats=1000, num_parallel_calls=12, ):
    tf_provider = make_datasetprovider(tf_records, repeats=repeats, num_parallel_calls=num_parallel_calls,
                                       batch_size=batch_size)
    sess = tf.Session()
    next_el = tf_provider.get_next()
    while True:
        try:
          yield sess.run(next_el)
        except tf.errors.OutOfRangeError:
            print("Iterations exhausted")
            break
            
def fetch_model():
    model = Sequential()
    model.add(Dense(2048, activation="relu", input_shape=(1024 + 128,)))
    model.add(Dense(3862, activation="sigmoid"))
    model.compile("adam", loss="binary_crossentropy")
    return model

train_data = "train0001.tfrecord"
train_data = glob("/home/venkatraman/Desktop/youtube-8m/data/video_level/trainGA.tfrecord")
#eval_data = glob("../input/video/train01.tfrecord")


#my_eval_iter = data_generator(eval_data)
model = fetch_model()
for epoch in tqdm(range(10)):
    for tdata in tqdm(train_data[:1000]):
        my_train_iter = data_generator(tdata, 64)
        model.fit_generator(my_train_iter, 
                            steps_per_epoch=30, 
                            epochs=1, verbose = 0)

