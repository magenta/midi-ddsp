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

"""The hyperparameters for training Synthesis Generator.
For details of each hyperparameters, please see the argpars """

class hparams:
  """Hyperparameters for training Synthesis Generator."""
  # Learning parameters
  batch_size = 4
  clip_grad = 2.5
  lr = 3e-4
  seed = 1111

  # Training parameters
  training_steps = 100000
  log_interval = 100
  checkpoint_save_interval = 5000
  eval_interval = 5000
  mode = 'train'  # train, eval
  data_dir = None
  restore_path = None

  # Model Parameters
  nhid = 256
  sequence_length = 1000
  train_synth_coder_first = True
  midi_audio_loss = True
  add_synth_loss = False
  synth_params_loss = False
  midi_decoder_type = 'interpretable_conditioning'
  position_code = 'index_length'
  midi_decoder_decoder_net = 'rnn_synth_params'
  multi_instrument = True
  instrument = 'vn'
  synth_coder_training_steps = 300
  lambda_recon = 1.0
  use_gan = True
  lambda_G = 1
  sg_z = True
  lr_disc = 1e-4
  write_tfrecord_audio = False
  without_note_expression = False
  discriminator_dim = 256

  # Synthesis & DSP parameters
  nhramonic = 60
  nnoise = 65
  reverb = True
  reverb_length = 48000
  num_mels = 64
  n_fft = 1024
  sample_rate = 16000
  frame_size = 64
  hop_length = frame_size
  win_length = hop_length * 2
  frame_shift_ms = 1000 / sample_rate * frame_size
  fmin = 40
