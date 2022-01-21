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

#!/bin/bash

batch_size=16
training_steps=50000
eval_interval=10000
checkpoint_save_interval=10000
synth_coder_training_steps=10000
data_dir=./data
midi_audio_loss=true
train_synth_coder_first=true
add_synth_loss=false
synth_params_loss=false
multi_instrument=true
midi_decoder_type=interpretable_conditioning
position_code=index_length
midi_decoder_decoder_net=rnn_synth_params
instrument=all
reverb=true
use_gan=true
lambda_recon=1.0
reverb_length=48000
name=logs_expression_generator

python train_synthesis_generator.py --batch_size $batch_size \
  --training_steps $training_steps \
  --data_dir $data_dir --name $name \
  --eval_interval $eval_interval \
  --checkpoint_save_interval $checkpoint_save_interval \
  --midi_audio_loss $midi_audio_loss \
  --train_synth_coder_first $train_synth_coder_first \
  --multi_instrument $multi_instrument \
  --midi_decoder_type $midi_decoder_type \
  --position_code $position_code \
  --instrument $instrument \
  --midi_decoder_decoder_net $midi_decoder_decoder_net \
  --add_synth_loss $add_synth_loss \
  --synth_params_loss $synth_params_loss \
  --reverb $reverb \
  --synth_coder_training_steps $synth_coder_training_steps \
  --use_gan $use_gan \
  --lambda_recon $lambda_recon \
  --reverb_length $reverb_length

synthesis_generator_weight_path=./logs/${name}/${training_steps}

python dump_expression_generator_dataset.py --model_path $synthesis_generator_weight_path \
  --data_dir $data_dir --output_dir ./logs/expression_generator_dataset

python train_expression_generator.py \
  --training_set_path ./logs/expression_generator_dataset/pickles/train_separate_piece.pickle \
  --test_set_path ./logs/expression_generator_dataset/pickles/test.pickle \
  --training_steps 5000
