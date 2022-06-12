"""Utility functions for inference."""
#  Copyright 2021 The DDSP Authors.
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
#
#  Lint as: python3

import ddsp
import numpy as np
import pandas as pd
import tensorflow as tf

# Note expression keys. This list is strictly ordered.
CONDITIONING_KEYS = ['volume',
                     'vol_fluc',
                     'vibrato',
                     'brightness',
                     'attack',
                     'vol_peak_pos']


def to_length(data, dst_length=1000, axis=1):
  """Make data to dst_length.
  Will pad 0 if have less length, and will cut off if have more length."""
  data_length = data.shape[axis]
  before_shape = data.shape[:axis]
  after_shape = data.shape[axis + 1:]
  if data_length < dst_length:
    pad_data = tf.zeros([*before_shape, dst_length - data_length, *after_shape],
                        dtype=data.dtype)
    return tf.concat([data, pad_data], axis)
  else:
    size = list(data.shape)
    size[axis] = dst_length
    return tf.slice(data, [0] * len(data.shape), size)


def ensure_same_length(data_all, axis=1):
  """Ensure the same length for all the data in the list."""
  max_length = max([d.shape[axis] for d in data_all])
  return [to_length(d, dst_length=max_length, axis=axis) for d in data_all]


def make_same_length(src, dst):
  """Make src the same length as the dst."""
  length = dst.shape[1]
  if src.shape[1] > length:
    return src[:, :length, ...]
  else:
    pad_shape = [src.shape[0], length - src.shape[1]] + src.shape[2:]
    return tf.concat([src, tf.zeros(pad_shape, dtype=src.dtype)], 1)


def conditioning_df_to_midi_features(conditioning_df, length=None):
  """Convert conditioning_df to midi features."""
  total_length = length if length is not None else \
    conditioning_df.tail(1)['offset'].values[0]
  q_pitch = np.zeros((1, total_length, 1), dtype=np.float32)
  onsets = np.zeros((1, total_length), dtype=np.int64)
  offsets = np.zeros((1, total_length), dtype=np.int64)
  for _, note in conditioning_df.iterrows():
    on = int(note['onset'])
    off = int(note['offset'])
    q_pitch[:, on:off + 1, :] = note['pitch']
    onsets[:, on] = 1
    offsets[:, off - 1] = 1

  # q_pitch, q_vel, f0_loss_weight, onsets, offsets
  midi_features = (q_pitch, q_pitch, q_pitch, onsets, offsets)
  return midi_features


def expression_generator_output_to_conditioning_df(expression_generator_outputs,
                                                   expression_generator_cond,
                                                   frame_size=0.004,
                                                   clip_value=True):
  """
  Convert expression generator outputs to conditioning_df.
  Args:
      expression_generator_outputs: The output of the expression generator.
      expression_generator_cond: (in dataset data['cond'])
      frame_size: the frame size, in second.
      clip_value: clip the expression generator output between [0, 1].

  Returns:

  """
  expression_generator_outputs = scale_expression_generator_output(
    expression_generator_outputs[0])
  if clip_value: # clip the expression generator output between [0, 1].
    expression_generator_outputs = tf.clip_by_value(
    expression_generator_outputs, clip_value_min=0.0, clip_value_max=1.0
    )
  conditioning_dict_keys = CONDITIONING_KEYS
  data_all = []
  onset = 0
  for i in range(expression_generator_outputs.shape[0]):
    pitch = expression_generator_cond['note_pitch'][0, i].numpy()
    length = int(round(expression_generator_cond['note_length'][
                         0, i, 0].numpy() / frame_size))
    offset = onset + length
    data_all.append(
      expression_generator_outputs[i].numpy().tolist() + [pitch, onset, offset,
                                                          length])
    onset += length
  columns = conditioning_dict_keys + ['pitch', 'onset', 'offset', 'note_length']
  conditioning_df = pd.DataFrame(data_all, columns=columns)
  return conditioning_df


def conditioning_df_to_expression_generator_output(conditioning_df,
                                                   instrument_id,
                                                   frame_size=0.004):
  """Convert conditioning_df to expression generator output."""
  note_pitch = conditioning_df['pitch'].values
  note_length = conditioning_df['note_length'].values

  note_pitch = tf.convert_to_tensor(note_pitch)[tf.newaxis, ...]
  note_length = tf.convert_to_tensor(note_length)[tf.newaxis, ...]
  note_length = tf.cast(note_length, tf.float32)[..., tf.newaxis] * frame_size

  note_sequence = {'note_pitch': note_pitch, 'note_length': note_length,
                   'instrument_id': tf.constant([instrument_id])}
  return note_sequence


def scale_expression_generator_output(output):
  """Scale expression generator output.
  This function now is useless. But it is a plcaeholder
  if you want to post-process your note expression output.
  """
  scale = np.ones((1, output.shape[-1]))
  scale = tf.convert_to_tensor(scale, tf.float32)
  output /= scale
  return output


def conditioning_df_to_audio(synthesis_generator, conditioning_df,
                             instrument_id, length=None,
                             display_progressbar=False):
  """Synthesize audio from conditioning_df."""
  conditioning_dict = conditioning_df_to_dict(conditioning_df, length=length)

  midi_features = conditioning_df_to_midi_features(conditioning_df,
                                                   length=length)
  midi_audio, midi_control_params, midi_synth_params = synthesis_generator. \
    gen_audio_from_cond_dict(conditioning_dict,
                             midi_features,
                             instrument_id=instrument_id,
                             display_progressbar=display_progressbar)

  return midi_audio, midi_control_params, midi_synth_params


def conditioning_dict_to_df(conditioning_dict, onsets, offsets, q_pitch):
  """Convert conditioning_dict to conditioning_df."""
  # Add onset to the beginning and offset to the end if needed.
  onsets, offsets, q_pitch = onsets.numpy(), offsets.numpy(), q_pitch.numpy()
  if q_pitch.ndim == 3:
    q_pitch = q_pitch[..., 0]
  onsets = np.where(onsets[0] == 1)[0]
  offsets = np.where(offsets[0] == 1)[0]
  if (len(onsets) == 0) or (
        q_pitch[0, 0] != 0 and onsets[0] != 0 and offsets[0] != 0):
    onsets = np.append(0, onsets)
  if (len(offsets) == 0) or (
        q_pitch[0, -1] != 0 and offsets[-1] != q_pitch.shape[1] and offsets[
    -1] != q_pitch.shape[1] - 1):
    offsets = np.append(offsets, q_pitch.shape[1])
  if offsets[0] == 0:
    offsets = offsets[1:]
  assert len(onsets) == len(offsets)

  # Make conditioning_df.
  data_all = []
  for on, off in zip(onsets.tolist(), offsets.tolist()):
    data_all.append(
      [v[0, on, 0].numpy() for v in conditioning_dict.values()] + [
        q_pitch[0, on], on, off,
        off - on])  # max(off-on, 1)
  columns = list(conditioning_dict.keys()) + ['pitch', 'onset', 'offset',
                                              'note_length']
  conditioning_df = pd.DataFrame(data_all, columns=columns)
  return conditioning_df


def conditioning_df_to_dict(conditioning_df, length=None):
  """Convert conditioning_df to conditioning_dict."""
  total_length = length if length is not None else \
    conditioning_df.tail(1)['offset'].values[0]
  conditioning_names = list(conditioning_df.columns)[:-4]
  conditioning_dict = {k: np.zeros((1, total_length, 1), dtype=np.float32) for k
                       in conditioning_names}
  for _, note in conditioning_df.iterrows():
    for name in conditioning_names:
      on = int(note['onset'])
      off = int(note['offset'])
      conditioning_dict[name][:, on:off + 1, :] = note[name]
  conditioning_dict = {k: tf.convert_to_tensor(v) for k, v in
                       conditioning_dict.items()}
  return conditioning_dict


def get_process_group(n_frames, frame_size=64, sample_rate=16000,
                      use_angular_cumsum=True):
  harmonic_synth = ddsp.synths.Harmonic(n_frames * frame_size, sample_rate,
                                        use_angular_cumsum=use_angular_cumsum)
  noise_synth = ddsp.synths.FilteredNoise(n_frames * frame_size, sample_rate)
  add = ddsp.processors.Add(name='add')
  # Create ProcessorGroup.
  dag = [(harmonic_synth, ['amplitudes', 'harmonic_distribution', 'f0_hz']),
         (noise_synth, ['noise_magnitudes']),
         (add, ['filtered_noise/signal', 'harmonic/signal'])]

  processor_group = ddsp.processors.ProcessorGroup(dag=dag,
                                                   name='processor_group')
  return processor_group
