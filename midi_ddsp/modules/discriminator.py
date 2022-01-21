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

"""The discriminator for Synthesis Generator."""

import tensorflow as tf

tfk = tf.keras
tfkl = tfk.layers

conv_init = tf.keras.initializers.RandomNormal(
  mean=0.0, stddev=0.02
)


class DBlock(tfkl.Layer):
  """Single discriminator block."""

  def __init__(self, nhid):
    super().__init__()
    self.cond_input_conv = tfkl.Conv1D(nhid, 3, 2, 'same',
                                       kernel_initializer=conv_init)
    self.spec_input_conv = tfkl.Conv1D(nhid, 3, 2, 'same',
                                       kernel_initializer=conv_init)

    self.cond_output_conv = tfkl.Conv1D(nhid, 3, 1, 'same',
                                        kernel_initializer=conv_init)
    self.spec_output_conv = tfkl.Conv1D(nhid, 3, 1, 'same',
                                        kernel_initializer=conv_init)

    self.leaky_relu = tfkl.LeakyReLU(0.2)

    self.cond_layer_norm = tfkl.LayerNormalization()
    self.spec_layer_norm = tfkl.LayerNormalization()

  def call(self, inputs):
    cond_features, spectral_features = inputs

    cond_z_in = self.cond_input_conv(cond_features)
    spec_z_in = self.spec_input_conv(spectral_features)
    spec_z_in += cond_z_in

    cond_z_out = self.leaky_relu(cond_z_in)
    spec_z_out = self.leaky_relu(spec_z_in)

    cond_z_out = self.cond_layer_norm(
      self.cond_output_conv(cond_z_out) + cond_z_in)
    spec_z_out = self.spec_layer_norm(
      self.spec_output_conv(spec_z_out) + spec_z_in)

    return cond_z_out, spec_z_out


class NLayerDiscriminator(tfkl.Layer):
  """A discriminator in a single level, containing multiple blocks."""

  def __init__(self, num_dblock=4, nhid=128):
    super().__init__()
    self.num_dblock = num_dblock
    self.dblock = {str(i): DBlock(nhid) for i in range(num_dblock)}
    self.out_proj = tfkl.Conv1D(1, 3, 1, 'same', kernel_initializer=conv_init)

  def call(self, inputs):
    feature_maps = []
    outputs = inputs
    for i in range(self.num_dblock):
      outputs = self.dblock[str(i)](outputs)
      feature_maps.append(outputs[1])

    out_score = self.out_proj(tf.concat(outputs, -1))
    feature_maps.append(out_score)

    return feature_maps


class Discriminator(tfkl.Layer):
  """The multi-scale discriminator, has num_D discriminators at num_D scale,
   each discriminator consists of num_dblock blocks."""

  def __init__(self, num_D=3, num_dblock=4, nhid=128):
    super().__init__()
    self.model = {}
    for i in range(num_D):
      self.model[f'disc_{i}'] = NLayerDiscriminator(
        num_dblock, nhid
      )

    self.downsample = tfkl.AveragePooling1D(4, 2, padding='same')

  def call(self, inputs):
    results = []
    for _, disc in self.model.items():
      results.append(disc(inputs))
      inputs = (self.downsample(inputs[0]), self.downsample(inputs[1]))
    return results
