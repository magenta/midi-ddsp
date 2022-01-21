#  Copyright 2022 The MIDI-DDSP Authors.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""The reverb modules used in MIDI-DDSP."""

import tensorflow as tf
import ddsp

tfk = tf.keras
tfkl = tfk.layers


def get_exp_decay(total_length, start_length, decay_exponent):
  """Get an exponential decay from 1 to 0.
  Args:
    total_length: Total length of the decay.
    start_length: The point where the decay starts.
    decay_exponent: The coefficient of the exponential decay.

  Returns: decay, the exponential decay from 1 to 0.

  """
  time = tf.linspace(0.0, 1.0, total_length - start_length)
  decay = tf.exp(-decay_exponent * time)
  decay = tf.concat([tf.ones(start_length), decay], 0)
  return decay


class ReverbModules(tfkl.Layer):
  """A reverb module for multi-instrument training.
  An unique reverb parameter is used for each instrument id.
  """

  def __init__(self, num_reverb=1, reverb_length=48000):
    super().__init__()
    self.num_reverb = num_reverb
    self.reverb_length = reverb_length
    self.reverb = ddsp.effects.Reverb(trainable=False,
                                      reverb_length=reverb_length)
    initializer = tf.random_normal_initializer(mean=0, stddev=1e-6)
    self.magnitudes_embedding = tfkl.Embedding(num_reverb, reverb_length,
                                               embeddings_initializer=
                                               initializer)

  def call(self, audio, reverb_number=0, training=False):
    batch_size = audio.shape[0]
    if isinstance(reverb_number, int):
      reverb_number = tf.repeat(tf.constant([reverb_number], dtype=tf.int64),
                                batch_size)
    if self.num_reverb == 1 or reverb_number is None:  # unified reverb
      reverb_number = tf.repeat(tf.constant([0], dtype=tf.int64), batch_size)
    ir_magnitudes = self.magnitudes_embedding(reverb_number)
    if not training:
      ir_magnitudes = ir_magnitudes * get_exp_decay(
        self.reverb_length, 16000, 4)[tf.newaxis, ...]
    output = self.reverb(audio, ir_magnitudes)
    return output
