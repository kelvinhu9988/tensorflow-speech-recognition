# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a trained audio graph against a WAVE file and reports the results.
"""

import argparse
import sys
import os
import glob
import csv
import time
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

FLAGS = None


def load_graph(filename):
  """Loads graph from file as the default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name):
  """Runs the audio data through the graph and prints predictions."""

  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions is a two-dimensional array, where one dimension
    #   represents the input image count, and the other has prediction scores per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions = sess.run(softmax_tensor, {input_layer_name: wav_data})[0]

    # Sort to show labels in order of confidence
    top_index = predictions.argsort()[-1]
    top_label = labels[top_index]
    top_score = predictions[top_index]
    return top_label, top_score, predictions


def label_wav(wav, labels_list, input_name, output_name):
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  prediction = run_graph(wav_data, labels_list, input_name, output_name)
  return prediction


def main(_):
  if not FLAGS.labels or not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('Labels file does not exist %s', FLAGS.labels)
  if not FLAGS.graph or not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('Graph file does not exist %s', FLAGS.graph)
  load_graph(FLAGS.graph)
  labels_list = load_labels(FLAGS.labels)

  search_path = os.path.join(FLAGS.data_dir, '*.wav')
  wav_path_list = glob.glob(search_path)
  wav_path_list.sort()
  threshold = 0.25

  with open('submission.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['fname', 'label'])
    for i in range(len(wav_path_list)):
      wav_path = wav_path_list[i]
      file_name = os.path.basename(wav_path)
      label, score, predictions = label_wav(wav_path, labels_list, FLAGS.input_name, FLAGS.output_name)
      if score < threshold and label != '_silence_' and label != '_unknown_':
        if predictions[labels_list.index('_silence_')] > predictions[labels_list.index('_unknown_')]:
          csvwriter.writerow([file_name, '_silence_'])
        else:
          csvwriter.writerow([file_name, '_unknown_'])
      else:
          csvwriter.writerow([file_name, label])
      if i % (len(wav_path_list) // 100) == 0:
          print('Complete: ' + str(int(i / (len(wav_path_list) - 1) * 100)) + '%', end='\r')

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--graph', type=str, default='', help='Model to use for identification.')
  parser.add_argument(
      '--labels', type=str, default='', help='Path to file containing labels.')
  parser.add_argument(
      '--data_dir', type=str, default='../test/audio', help='Folder of audio files to be identified.')
  parser.add_argument(
      '--input_name',
      type=str,
      default='wav_data:0',
      help='Name of WAVE data input node in the model.')
  parser.add_argument(
      '--output_name',
      type=str,
      default='labels_softmax:0',
      help='Name of node outputting a prediction in the model.')


  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
