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

"""Training code for Synthesis Generator."""

import tensorflow as tf
import time
import os
import sys
import logging
import argparse

from midi_ddsp.data_handling.get_dataset import get_dataset
from midi_ddsp.utils.training_utils import print_hparams, set_seed, \
  save_results, str2bool
from midi_ddsp.utils.summary_utils import write_tensorboard_audio
from midi_ddsp.hparams_synthesis_generator import hparams as hp
from midi_ddsp.modules.recon_loss import ReconLossHelper
from midi_ddsp.modules.gan_loss import GANLossHelper
from midi_ddsp.modules.get_synthesis_generator import get_synthesis_generator, \
  get_fake_data_synthesis_generator
from midi_ddsp.modules.discriminator import Discriminator

parser = argparse.ArgumentParser(description='Train Synthesis Generator.')

set_seed(hp.seed)


def train(training_data, training_steps, start_step=1):
  """Training loop including evaluation."""
  start_time = time.time()
  loss_helper.reset_metrics()

  for step in range(start_step, training_steps + start_step + 1):
    data = next(training_data)

    # Run the model and get the loss.
    with tf.GradientTape() as tape, tf.GradientTape() as disc_tape:
      outputs = model(data, training=True,
                      run_synth_coder_only=hp.run_synth_coder_only)
      loss_dict_recon = loss_helper.compute_loss(data, outputs,
                                                 synth_coder_only=
                                                 hp.run_synth_coder_only,
                                                 add_synth_loss=
                                                 hp.add_synth_loss)
      if not hp.run_synth_coder_only and hp.use_gan:
        cond, real_outputs, fake_outputs = gan_loss_helper.get_disc_input(
          outputs)
        D_fake = net_D([cond, fake_outputs])
        D_real = net_D([cond, real_outputs])
        loss_dict_disc = gan_loss_helper.compute_disc_loss(D_fake, D_real)
        loss, loss_dict_gen = gan_loss_helper.compute_gen_loss(D_fake, D_real,
                                                               loss_dict_recon[
                                                                 'total_loss'])
      else:
        loss = loss_dict_recon['total_loss']

    # Clip and apply gradients.
    grads = tape.gradient(loss, model.trainable_variables)
    grads, _ = tf.clip_by_global_norm(grads, hp.clip_grad)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    loss_helper.update_metrics(loss_dict_recon)
    loss_helper.write_summary(loss_dict_recon, writer, 'Train', step)

    # Train discriminator and update GAN loss.
    if not hp.run_synth_coder_only and hp.use_gan:
      gradients_of_discriminator = disc_tape.gradient(
        loss_dict_disc['disc_loss'], net_D.trainable_variables)
      optimizer_disc.apply_gradients(
        zip(gradients_of_discriminator, net_D.trainable_variables))
      gan_loss_helper.update_metrics(loss_dict_disc)
      gan_loss_helper.write_summary(loss_dict_disc, writer, 'Train', step)
      gan_loss_helper.update_metrics(loss_dict_gen)
      gan_loss_helper.write_summary(loss_dict_gen, writer, 'Train', step)

    # Print logging summary.
    if step % hp.log_interval == 0:
      elapsed = time.time() - start_time
      current_lr = optimizer._decayed_lr('float32').numpy()
      msg = f'| {step:6d} steps | lr {current_lr:02.2e} ' \
            f'| ms/batch {(elapsed * 1000 / hp.log_interval):5.2f} '
      msg = msg + loss_helper.get_loss_log()
      loss_helper.reset_metrics()
      if not hp.run_synth_coder_only:
        msg = msg + gan_loss_helper.get_loss_log()
        gan_loss_helper.reset_metrics()
      logging.info(msg)
      start_time = time.time()

    # Evaluate.
    if step % hp.eval_interval == 0:
      evaluate(evaluation_data, step)

      # Synthesize training data.
      outputs = model(train_sample_batch, training=True,
                      run_synth_coder_only=hp.run_synth_coder_only)
      save_results(outputs['synth_audio'], train_sample_batch['audio'], log_dir,
                   f'train_{step}_synth',
                   hp.sample_rate)
      if 'midi_audio' in outputs.keys():
        save_results(outputs['midi_audio'], train_sample_batch['audio'],
                     log_dir, f'train_{step}_midi',
                     hp.sample_rate)
      if hp.write_tfrecord_audio:
        write_tensorboard_audio(writer, train_sample_batch, outputs, step,
                                tag='Train')

      # Synthesize evaluation data.
      outputs = model(eval_sample_batch, training=False,
                      run_synth_coder_only=hp.run_synth_coder_only)
      save_results(outputs['synth_audio'], eval_sample_batch['audio'], log_dir,
                   f'eval_{step}_synth',
                   hp.sample_rate)
      if 'midi_audio' in outputs.keys():
        save_results(outputs['midi_audio'], eval_sample_batch['audio'], log_dir,
                     f'eval_{step}_midi',
                     hp.sample_rate)
      if hp.write_tfrecord_audio:
        write_tensorboard_audio(writer, eval_sample_batch, outputs, step,
                                tag='Eval')

    # DDSP Inference training finished.
    # Start training Synthesis Generator and
    # dump dataset for expression generator.
    if (step - start_step + 1) == hp.synth_coder_training_steps:
      hp.run_synth_coder_only = False
      if not hp.add_synth_loss:
        model.freeze_synth_coder()

    # Save weights for the whole model.
    if step % hp.checkpoint_save_interval == 0:
      model.save_weights(f'{log_dir}/{step}')


def evaluate(evaluation_data, step):
  """Evaluating the test set."""
  eval_loss_helper = ReconLossHelper(hp, eval_recon_loss=True)
  start_time = time.time()
  for data in evaluation_data:
    outputs = model(data, training=False,
                    run_synth_coder_only=hp.run_synth_coder_only)

    loss_dict = eval_loss_helper.compute_loss(data, outputs,
                                              synth_coder_only=
                                              hp.run_synth_coder_only)
    eval_loss_helper.update_metrics(loss_dict)

  eval_loss_helper.write_mean_summary(writer, 'Eval', step)
  msg = f'eval: | step {step:6d} | eval time: {(time.time() - start_time):3.3f}'
  msg = msg + eval_loss_helper.get_loss_log()
  logging.info(msg)


if __name__ == '__main__':
  parser.add_argument('--batch_size', type=int, default=hp.batch_size,
                      help='Batch size to use for training.')
  parser.add_argument('--nhid', type=int, default=hp.nhid,
                      help='Number of hidden dimensions in '
                           'Synthesis Generator.')
  parser.add_argument('--training_steps', type=int, default=hp.training_steps,
                      help='Number of training steps to take.')
  parser.add_argument('--eval_interval', type=int, default=hp.eval_interval,
                      help='The number of training steps to take '
                           'evaluation on whole evaluation set.')
  parser.add_argument('--checkpoint_save_interval', type=int,
                      default=hp.checkpoint_save_interval,
                      help='The number of training steps to take before save '
                           'the model weights once.')
  parser.add_argument('--data_dir', type=str, default=hp.data_dir,
                      help='The directory to tfrecord data files.')
  parser.add_argument('--restore_path', type=str, default=hp.restore_path,
                      help='The path to the model weights file for restore '
                           'training.')
  parser.add_argument('--midi_audio_loss', type=str2bool, nargs='?', const=True,
                      default=hp.midi_audio_loss,
                      help='Whether to use '
                           'multi-scale spectral loss on audio predicted')
  parser.add_argument('--synth_params_loss', type=str2bool, nargs='?',
                      const=True, default=hp.synth_params_loss,
                      help='Whether to use L1 loss on synthesis parameters '
                           'predicted.')
  parser.add_argument('--train_synth_coder_first', type=str2bool, nargs='?',
                      const=True, default=hp.train_synth_coder_first,
                      help='Whether to train DDSP Inference first or joint '
                           'training DDSP Inference and Synthesis Generator.')
  parser.add_argument('--add_synth_loss', type=str2bool, nargs='?', const=True,
                      default=hp.add_synth_loss,
                      help='Whether to add DDSP Inference loss to the '
                           'total loss for back-prop.')
  parser.add_argument('--multi_instrument', type=str2bool, nargs='?',
                      const=True, default=hp.multi_instrument,
                      help='Whether to train multi-instrument model or '
                           'single-instrument model')
  parser.add_argument('--position_code', type=str, default=hp.position_code,
                      help='Whether to use positional encoding in '
                           'Synthesis Generator, and what type of '
                           'positional encoding.')
  parser.add_argument('--midi_decoder_type', type=str,
                      default=hp.midi_decoder_type,
                      help='The type of Synthesis Generator.')
  parser.add_argument('--midi_decoder_decoder_net', type=str,
                      default=hp.midi_decoder_decoder_net,
                      help='The network type for Synthesis Generator.')
  parser.add_argument('--reverb', type=str2bool, nargs='?', const=True,
                      default=hp.reverb, help='Whether to use reverb.')
  parser.add_argument('--reverb_length', type=int, default=hp.reverb_length,
                      help='The length for the reverb impulse response.')
  parser.add_argument('--instrument', type=str, default=hp.instrument,
                      help='The instrument to train on for single instrument '
                           'setting.')
  parser.add_argument('--synth_coder_training_steps', type=int,
                      default=hp.synth_coder_training_steps,
                      help='The number of steps for training DDSP Inference.')
  parser.add_argument('--use_gan', type=str2bool, nargs='?', const=True,
                      default=hp.use_gan, help='Whether to use GAN training '
                                               'for Synthesis Generator.')
  parser.add_argument('--lambda_recon', type=float, default=hp.lambda_recon,
                      help='The weight for the reconstruction loss.')
  parser.add_argument('--lambda_G', type=float, default=hp.lambda_G,
                      help='The weight for the discriminator loss '
                           'for generator.')
  parser.add_argument('--sg_z', type=str2bool, nargs='?', const=True,
                      default=hp.sg_z, help='Whether to stop the gradient for'
                                            'conditioning sequence (z) in '
                                            'discriminator.')
  parser.add_argument('--lr_disc', type=float, default=hp.lr_disc,
                      help='The learning rate for discriminator.')
  parser.add_argument('--without_note_expression', type=str2bool, nargs='?',
                      const=True, default=hp.without_note_expression,
                      help='Whether to train Synthesis Generator using no note '
                           'expression controls. Used in paper rebuttal.')
  parser.add_argument('--mode', type=str, default=hp.mode,
                      help='The mode to run, train or eval.')
  parser.add_argument('--name', type=str, default='logs_synthesis_generator',
                      help='The name of the log folder.')

  # Change hp according to argparse.
  args = parser.parse_args()
  for k, v in vars(args).items():
    setattr(hp, k, v)

  # Create Synthesis Generator
  model = get_synthesis_generator(hp)
  model._build(get_fake_data_synthesis_generator(hp))

  # Create optimizer, loss helper and discriminator.
  scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=hp.lr, decay_steps=1000,
    decay_rate=0.99)
  optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)

  loss_helper = ReconLossHelper(hp)
  gan_loss_helper = GANLossHelper(lambda_recon=hp.lambda_recon,
                                  lambda_G=hp.lambda_G, sg_z=hp.sg_z)
  optimizer_disc = tf.keras.optimizers.Adam(learning_rate=hp.lr_disc)

  net_D = Discriminator(nhid=hp.discriminator_dim)
  # 64=instrument_emb_dim
  z_dim = hp.discriminator_dim + int(hp.multi_instrument) * 64
  # synth_params_dim = dim(nharmonic + nnoise + amplitude + f0)
  synth_params_dim = hp.nhramonic + hp.nnoise + 2
  _ = net_D((tf.random.normal([4, 1000, z_dim]),
             tf.random.normal([4, 1000, synth_params_dim])))

  # Load model, create log directory and log file.
  log_dir = f'logs/{args.name}'

  if hp.restore_path:
    model.load_weights(hp.restore_path)
    log_dir = os.path.dirname(hp.restore_path)

  writer = tf.summary.create_file_writer(log_dir)
  log_path = os.path.join(log_dir, 'train.log')
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s - %(levelname)s: %(message)s',
                      handlers=[
                        logging.FileHandler(log_path),
                        logging.StreamHandler(sys.stdout)]
                      )

  # Load dataset.
  training_data, length_training_data, \
  evaluation_data, length_evaluation_data = get_dataset(hp)
  eval_sample_batch = next(iter(evaluation_data))
  train_sample_batch = next(training_data)
  logging.info('Data loaded! Data size: %s', str(length_training_data))

  # Print model summary and hyperparameters.
  model.summary(print_fn=logging.info)
  logging.info(str(print_hparams(hp)))

  # Start training loop
  start_step = int(os.path.basename(hp.restore_path)) if hp.restore_path else 1

  if hp.mode == 'train':
    if hp.train_synth_coder_first:
      hp.run_synth_coder_only = True
      model.train_synth_coder_only()
    else:
      hp.run_synth_coder_only = False

    train(training_data, hp.training_steps, start_step=start_step)

  elif hp.mode == 'eval':
    hp.run_synth_coder_only = False
    evaluate(evaluation_data, start_step)
