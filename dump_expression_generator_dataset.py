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

"""Dump the dataset for training expression generator."""

import os
import argparse
from midi_ddsp.utils.training_utils import set_seed, get_hp
from midi_ddsp.utils.create_expression_generator_dataset_utils import \
  dump_expression_generator_dataset
from midi_ddsp.hparams_synthesis_generator import hparams as hp
from midi_ddsp.modules.get_synthesis_generator import get_synthesis_generator, \
  get_fake_data_synthesis_generator

parser = argparse.ArgumentParser(description='Dump expression generator '
                                             'dataset.')
set_seed(1234)


def main():
  parser.add_argument('--model_path', type=str,
                      default=None,
                      help='The path to the model checkpoint.')
  parser.add_argument('--data_dir', type=str,
                      default=None,
                      help='The directory to the unbatched tfrecord dataset.')
  parser.add_argument('--output_dir', type=str,
                      default=None,
                      help='The output directory for dumping the expression '
                           'generator dataset.')
  # TODO: (yusongwu) add automatic note expression scaling
  args = parser.parse_args()
  model_path = args.model_path
  hp_dict = get_hp(os.path.join(os.path.dirname(model_path), 'train.log'))
  for k, v in hp_dict.items():
    setattr(hp, k, v)
  model = get_synthesis_generator(hp)
  model._build(get_fake_data_synthesis_generator(hp))
  model.load_weights(model_path)

  print('Creating dataset for expression generator!')
  dump_expression_generator_dataset(model, data_dir=args.data_dir,
                                    output_dir=args.output_dir)


if __name__ == '__main__':
  main()
