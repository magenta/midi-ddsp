"""Utility functions for audio input, output and spectral operations."""

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

import librosa
import librosa.display
import librosa.filters
import numpy as np
import soundfile
import crepe
import tensorflow as tf
import ddsp
from hmmlearn import hmm


def save_wav(wav, path, sample_rate=16000):
  soundfile.write(path, wav, sample_rate)


def load_audio(file_path, sample_rate, mono=True, window_size=0,
               from_numpy=False):
  """
  Load audio file from disk.
  :param file_path: Audio file path.
  :param sample_rate: Sample rate to load, will resample to the sample_rate
  if file has a different sample rate.
  :param mono: Whether to load a mono audio file.
  :param window_size: If larger than 0, then the waveform will be cut off to
  have a length that is multiple of window_size,
  :param from_numpy: If the file_path provided is pointing to a npy file.
  :return:
  """
  if from_numpy:
    y = np.load(file_path)
  else:
    y, _ = librosa.load(file_path, sample_rate, mono=mono, dtype=np.float64)
  if window_size > 0:
    output_length = len(y) // window_size * window_size
    y = y[:output_length]
  return y


def extract_f0(wav, frame_shift_ms=5, sr=44100, unvoice=True, no_log=False):
  """Extract f0 from audio using CREPE."""
  if sr != 16000:
    raise RuntimeError('CREPE method should use sr=16khz')
  _, frequency, confidence, _ = crepe.predict(
    wav, sr=sr,
    viterbi=True,
    step_size=frame_shift_ms,
    verbose=0 if no_log else 1)
  f0 = frequency
  if unvoice:
    is_voiced = predict_voicing(confidence)
    frequency_unvoiced = frequency * is_voiced
    f0 = frequency_unvoiced

  return f0


def predict_voicing(confidence):
  # https://github.com/marl/crepe/pull/26
  """
  Find the Viterbi path for voiced versus unvoiced frames.
  Parameters
  ----------
  confidence : np.ndarray [shape=(N,)]
      voicing confidence array, i.e. the confidence in the presence of
      a pitch
  Returns
  -------
  voicing_states : np.ndarray [shape=(N,)]
      HMM predictions for each frames state, 0 if unvoiced, 1 if
      voiced
  """
  # uniform prior on the voicing confidence
  starting = np.array([0.5, 0.5])

  # transition probabilities inducing continuous voicing state
  transition = np.array([[0.99, 0.01], [0.01, 0.99]])

  # mean and variance for unvoiced and voiced states
  means = np.array([[0.0], [1.0]])
  variances = np.array([[0.25], [0.25]])

  # fix the model parameters because we are not optimizing the model
  model = hmm.GaussianHMM(n_components=2)
  model.startprob_, model.covars_, model.transmat_, model.means_, \
  model.n_features = starting, variances, transition, means, 1

  # find the Viterbi path
  voicing_states = model.predict(confidence.reshape(-1, 1), [len(confidence)])

  return np.array(voicing_states)


def spectral_centroid(wav, hop_length, sr):
  centroid = librosa.feature.spectral_centroid(y=wav, sr=sr,
                                               hop_length=hop_length)
  return centroid


def tf_stft(audio, win_length, hop_length, n_fft, pad_end=True):
  s = tf.signal.stft(
    signals=audio,
    frame_length=win_length,
    frame_step=hop_length,
    fft_length=n_fft,
    pad_end=pad_end)
  mag = tf.abs(s)
  return tf.cast(mag, tf.float32)


def tf_mel(audio, sample_rate, win_length, hop_length, n_fft, num_mels, fmin=40,
           pad_end=True):
  """Calculate Mel Spectrogram."""
  mag = tf_stft(audio, win_length, hop_length, n_fft, pad_end=pad_end)
  num_spectrogram_bins = int(mag.shape[-1])
  hi_hz = sample_rate // 2
  linear_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mels,
    num_spectrogram_bins,
    sample_rate,
    fmin,
    hi_hz)
  mel = tf.tensordot(mag, linear_to_mel_matrix, 1)
  mel.set_shape(mag.shape[:-1].concatenate(linear_to_mel_matrix.shape[-1:]))
  return mel


def tf_log_mel(audio, sample_rate, win_length, hop_length, n_fft, num_mels,
               fmin=40, pad_end=True):
  mel = tf_mel(audio, sample_rate, win_length, hop_length, n_fft, num_mels,
               fmin=fmin, pad_end=pad_end)
  return ddsp.core.safe_log(mel)
