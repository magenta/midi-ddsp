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

"""Synthesize any MIDI file using MIDI-DDSP through command line."""

# Ignore a bunch of warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pretty_midi
import argparse
import glob
from tqdm.autonotebook import tqdm

from midi_ddsp.data_handling.instrument_name_utils import \
  INST_NAME_TO_MIDI_PROGRAM_DICT, MIDI_PROGRAM_TO_INST_ID_DICT, \
  MIDI_PROGRAM_TO_INST_NAME_DICT
from midi_ddsp.utils.midi_synthesis_utils import note_list_to_sequence, \
  expression_generator_output_to_conditioning_df, batch_conditioning_df_to_audio
from midi_ddsp.utils.audio_io import save_wav
from midi_ddsp.utils.training_utils import get_hp
from midi_ddsp.utils.inference_utils import ensure_same_length
from midi_ddsp.utils.file_utils import pickle_dump
from midi_ddsp.hparams_synthesis_generator import hparams as hp
from midi_ddsp.modules.get_synthesis_generator import get_synthesis_generator, \
  get_fake_data_synthesis_generator
from midi_ddsp.modules.expression_generator import ExpressionGenerator, \
  get_fake_data_expression_generator

FRAME_RATE = 250
SAMPLE_RATE = 16000


def load_pretrained_model(synthesis_generator_path=None,
                          expression_generator_path=None):
  """Load pre-trained model weights."""
  package_dir = os.path.dirname(os.path.realpath(__file__))

  if not os.path.exists(os.path.join(package_dir,
                                     'midi_ddsp_model_weights_urmp_9_10')):
    raise FileNotFoundError('Model weights not found. '
                            'Please run \'midi_ddsp_download_model_weights\' '
                            'to download model weights, '
                            'or specify path to model weights.')

  if synthesis_generator_path is None:
    synthesis_generator_path = os.path.join(
      package_dir,
      'midi_ddsp_model_weights_urmp_9_10',
      'synthesis_generator',
      '50000')
  if expression_generator_path is None:
    expression_generator_path = os.path.join(
      package_dir,
      'midi_ddsp_model_weights_urmp_9_10',
      'expression_generator',
      '5000')

  hp_dict = get_hp(
    os.path.join(os.path.dirname(synthesis_generator_path), 'train.log'))
  for k, v in hp_dict.items():
    setattr(hp, k, v)
  synthesis_generator = get_synthesis_generator(hp)
  synthesis_generator._build(get_fake_data_synthesis_generator(hp))
  synthesis_generator.load_weights(synthesis_generator_path).expect_partial()

  n_out = 6
  expression_generator = ExpressionGenerator(n_out=n_out, nhid=128)
  fake_data = get_fake_data_expression_generator(n_out)
  _ = expression_generator(fake_data['cond'], out=fake_data['target'],
                           training=True)
  expression_generator.load_weights(expression_generator_path).expect_partial()
  return synthesis_generator, expression_generator


def synthesize_midi(synthesis_generator, expression_generator, midi_file,
                    pitch_offset=0, speed_rate=1.0,
                    output_dir=None,
                    use_fluidsynth=False,
                    sf2_path='/usr/share/sounds/sf2/FluidR3_GM.sf2',
                    display_progressbar=True,
                    skip_existing_files=False,
                    save_metadata=False):
  """
  Synthesize a midi file using MIDI-DDSP.
  Args:
      synthesis_generator: The instance of a Synthesis Generator.
      expression_generator: The instance of a expression generator.
      midi_file: The path to the MIDI file.
      pitch_offset: Pitch in semitone to transpose.
      speed_rate: The speed to synthesize the MIDI file.
      output_dir: The directory for saving outputs.
        If output_dir is None, outputs will not be written to disk.
      use_fluidsynth: Whether to use FluidSynth for synthesizing instruments
        that are not available in MIDI-DDSP.
      sf2_path: The path to a sf2 soundfont file used for FluidSynth.
      display_progressbar: Whether to display progress bar.
      skip_existing_files: Skip synthesizing MIDI files if already exist
        output folders.
      save_metadata: also save metadata containing instrument_id, note
        expression controls, and synthesis parameters.

  Returns: A dict of output:
          'mix_audio': mix audio,
          'stem_audio': stem audios,
          'part_synth_by_model': list of part numbers that are synthesized
            by MIDI-DDSP,
          'midi_control_params': control parameters generated by MIDI-DDSP,
          'midi_synth_params': synth parameters generated by MIDI-DDSP,
          'conditioning_df': note expressions predicted by expression generator
            in the format of DataFrame,
  """
  # Check if there is existing files.
  filename = os.path.splitext(os.path.basename(midi_file))[0]
  if output_dir is not None:
    output_dir = os.path.join(output_dir, filename)
    if os.path.exists(output_dir) and skip_existing_files:
      print(f'{midi_file} has been synthesized, will skip this file.')
      return

  # Get all the midi program in URMP dataset excluding guitar.
  allowed_midi_program = list(INST_NAME_TO_MIDI_PROGRAM_DICT.values())[:-1]
  midi_data = pretty_midi.PrettyMIDI(midi_file)
  instrument_id_all = []
  conditioning_df_all = []
  part_synth_by_model = []
  midi_audio_all = {}
  midi_synth_params_all = {}
  midi_control_params_all = {}

  # For each part, predict expressions using MIDI-DDSP,
  # or synthesize using FluidSynth.
  for part_number, instrument in enumerate(midi_data.instruments):
    midi_program = instrument.program
    if midi_program in allowed_midi_program:
      note_sequence = note_list_to_sequence(instrument.notes, fs=FRAME_RATE,
                                            pitch_offset=pitch_offset,
                                            speed_rate=speed_rate)
      instrument_id = tf.constant([MIDI_PROGRAM_TO_INST_ID_DICT[midi_program]])
      instrument_id_all.append(instrument_id)
      note_sequence['instrument_id'] = instrument_id
      expression_generator_outputs = expression_generator(note_sequence,
                                                          out=None,
                                                          training=False)
      conditioning_df = expression_generator_output_to_conditioning_df(
        expression_generator_outputs['output'], note_sequence)
      conditioning_df_all.append(conditioning_df)
      part_synth_by_model.append(part_number)
    elif use_fluidsynth:
      instrument_name = MIDI_PROGRAM_TO_INST_NAME_DICT[midi_program]
      print(
        f'Part {part_number} in {midi_file} has {instrument_name} as '
        f'instrument which cannot be synthesized by model. '
        f'Using fluidsynth instead.')

      fluidsynth_wav_r3 = instrument.fluidsynth(SAMPLE_RATE, sf2_path=sf2_path)
      fluidsynth_wav_r3 *= 0.25  # * 0.25 for lower volume
      midi_audio_all[part_number] = fluidsynth_wav_r3

  # Synthesize audio in batch using Synthesis Generator.
  if len(conditioning_df_all) > 0:
    midi_audio, midi_control_params, midi_synth_params = \
      batch_conditioning_df_to_audio(
        synthesis_generator,
        conditioning_df_all,
        instrument_id_all,
        display_progressbar=display_progressbar)

    # discard rest of the values in midi_synth_params other than inputs
    midi_synth_params = midi_synth_params['inputs']
    conditioning_df_all_for_save = {}
    instrument_id_all_for_save = {}
    for i in range(midi_audio.shape[0]):
      part_number = part_synth_by_model[i]

      # align audio with part number
      midi_audio_all[part_number] = midi_audio[i].numpy()

      # align instrument_id with part number
      instrument_id_all_for_save[part_number] = instrument_id_all[i].numpy()[0]

      # align note expression controls with part number
      conditioning_df_all_for_save[part_number] = conditioning_df_all[i]

      # align synthesis parameters with part number
      # get the midi synth parameters
      midi_synth_params_all[part_number] = {
        k: v[i].numpy() for k, v in midi_synth_params.items()}

      # align control parameters with part number
      # (yusongwu) sorry for mis-aligned variable names between
      # synth_params and control_params due to historical issues
      midi_control_params_all[part_number] = {
        'amplitudes': midi_control_params[1][i].numpy(),
        'harmonic_distribution': midi_control_params[2][i].numpy(),
        'noise_magnitudes': midi_control_params[3][i].numpy(),
        'f0_hz': midi_control_params[0][i].numpy(),
      }

  # Sorting out and save the wav.
  if midi_audio_all:

    # If there is audio synthesized, mix the audio and return the output.
    midi_audio_mix = np.sum(
      np.stack(ensure_same_length(
        [a.astype(np.float64) for a in midi_audio_all.values()], axis=0),
        axis=-1),
      axis=-1)

    output = {
      'mix_audio': midi_audio_mix,
      'stem_audio': midi_audio_all,
      'part_synth_by_model': part_synth_by_model,
      'midi_control_params': midi_control_params_all,
      'midi_synth_params': midi_synth_params_all,
      'conditioning_df': conditioning_df_all,
    }

    # if provided with output_dir, save all the files needed
    if output_dir is not None:
      os.makedirs(output_dir, exist_ok=True)
      for part_number, instrument in enumerate(midi_data.instruments):
        midi_program = instrument.program
        instrument_name = MIDI_PROGRAM_TO_INST_NAME_DICT[midi_program]
        if midi_program in allowed_midi_program:
          audio = midi_audio_all[part_number]
          save_wav(audio,
                   os.path.join(
                     output_dir,
                     f'{part_number}_{instrument_name}.wav'),
                   SAMPLE_RATE)
        elif use_fluidsynth:
          audio = midi_audio_all[part_number]
          save_wav(audio,
                   os.path.join(
                     output_dir,
                     f'{part_number}_{instrument_name}_fluidsynth.wav'),
                   SAMPLE_RATE)

      save_wav(midi_audio_mix, os.path.join(output_dir, 'mix.wav'), SAMPLE_RATE)
      if save_metadata:
        metadata = {
            'instrument_id': instrument_id_all_for_save,
            'note_expression_control': conditioning_df_all_for_save,
            'synthesis_parameters': midi_synth_params_all,
        }
        pickle_dump(metadata, os.path.join(output_dir, 'metadata.pickle'))
  else:
    output = None

  return output


def main():
  parser = argparse.ArgumentParser(
    description='Synthesize MIDI files using MIDI-DDSP.')
  parser.add_argument('--midi_dir', type=str, default=None,
                      help='The directory containing MIDI files.')
  parser.add_argument('--midi_path', type=str, default=None,
                      help='The path to a MIDI file.')
  parser.add_argument('--pitch_offset', type=int, default=0,
                      help='Pitch offset to transpose in semitone.')
  parser.add_argument('--speed_rate', type=float, default=1.0,
                      help='The speed to synthesize the MIDI file(s).')
  parser.add_argument('--sf2_path', type=str,
                      default=None,
                      help='The path to a sf2 soundfont file.')
  parser.add_argument('--output_dir', type=str, default=None,
                      help='The directory for output audio.')
  parser.add_argument('--use_fluidsynth', action='store_true',
                      help='Use FluidSynth to synthesize the midi instruments '
                           'that are not contained in MIDI-DDSP.')
  parser.add_argument('--synthesis_generator_weight_path', type=str,
                      default=None,
                      help='The path to the Synthesis Generator weights. '
                           'It is not a specific file path but an index path. '
                           'See https://www.tensorflow.org/guide/checkpoint#'
                           'restore_and_continue_training.')
  parser.add_argument('--expression_generator_weight_path', type=str,
                      default=None,
                      help='The path to the expression generator weights. '
                           'It is not a specific file path but an index path. '
                           'See https://www.tensorflow.org/guide/checkpoint#'
                           'restore_and_continue_training.')
  parser.add_argument('--skip_existing_files', dest='skip_existing_files',
                      action='store_true',
                      help='Skip synthesizing MIDI files if already exist '
                           'output folders.')
  parser.add_argument('--save_metadata', dest='save_metadata',
                      action='store_true',
                      help='Save metadata including containing instrument_id, '
                           'note expression controls, and synthesis '
                           'parameters generated by MIDI-DDSP.')

  args = parser.parse_args()

  synthesis_generator, expression_generator = load_pretrained_model(
    synthesis_generator_path=args.synthesis_generator_weight_path,
    expression_generator_path=args.expression_generator_weight_path)

  if args.output_dir is None:
    print('Output directory not specified. Output to current directory.')
    output_dir = './'
  else:
    output_dir = args.output_dir

  if args.midi_dir and args.midi_path:
    print('Both midi_dir and midi_path are provided. Will use midi_dir.')
  elif not args.midi_dir and not args.midi_path:
    raise ValueError('None of midi_dir or midi_path is provided. '
                     'Please provide at least one of midi_dir or midi_path.')
  elif args.midi_dir:
    midi_file_list = glob.glob(args.midi_dir + '/*.mid')
    if len(midi_file_list) == 0:
      raise FileNotFoundError('No midi files found in the directory.')
    for midi_file in tqdm(midi_file_list, desc='Generating files: '):
      synthesize_midi(
        synthesis_generator,
        expression_generator,
        midi_file,
        pitch_offset=args.pitch_offset,
        speed_rate=args.speed_rate,
        output_dir=output_dir,
        sf2_path=args.sf2_path,
        use_fluidsynth=args.use_fluidsynth,
        display_progressbar=False,
        skip_existing_files=args.skip_existing_files,
        save_metadata=args.save_metadata,
      )
  elif args.midi_path:
    synthesize_midi(
      synthesis_generator,
      expression_generator,
      args.midi_path,
      pitch_offset=args.pitch_offset,
      speed_rate=args.speed_rate,
      output_dir=output_dir,
      sf2_path=args.sf2_path,
      use_fluidsynth=args.use_fluidsynth,
      display_progressbar=True,
      skip_existing_files=args.skip_existing_files,
      save_metadata=args.save_metadata,
    )


if __name__ == '__main__':
  main()
