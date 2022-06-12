<div align="center">
<img src="https://midi-ddsp.github.io/pics/midi-ddsp-logo.png" width="200px" alt="logo"></img>
</div>

# MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling

[Demos](https://midi-ddsp.github.io/) | [Blog Post](https://magenta.tensorflow.org/midi-ddsp) 
| [Colab Notebook](https://colab.research.google.com/github/magenta/midi-ddsp/blob/main/midi_ddsp/colab/MIDI_DDSP_Demo.ipynb)
| [Paper](https://arxiv.org/abs/2112.09312) |  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/lukewys/midi-ddsp)


MIDI-DDSP is a hierarchical audio generation model for synthesizing MIDI expanded
from [DDSP](https://github.com/magenta/ddsp).

## Links

* [Check out the blog post 💻](https://magenta.tensorflow.org/midi-ddsp)
* [Read the original paper 📄](https://arxiv.org/abs/2112.09312)
* [Listen to some examples 🔈](https://midi-ddsp.github.io/)
* [Try out MIDI Synthesis using MIDI-DDSP 🎵->🎻🔊](https://colab.research.google.com/github/magenta/midi-ddsp/blob/main/midi_ddsp/colab/MIDI_DDSP_Demo.ipynb)
* [Try out Web Demo on Huggingface Spaces](https://huggingface.co/spaces/akhaliq/midi-ddsp)


## Install MIDI-DDSP

You could install MIDI-DDSP via pip, which allows you to use the
cool [Command-line MIDI synthesis](#command-line-midi-synthesis) to synthesize your MIDI.

To install MIDI-DDSP via pip, simply run following command in a **python3.8** environment:

```
pip install midi-ddsp
```

MIDI-DDSP is developed using tensorflow 2.7.0. Newer tensorflow version should also work.

## Train MIDI-DDSP

To train MIDI-DDSP, please first install midi-ddsp and clone the MIDI-DDSP repository:

```
git clone https://github.com/magenta/midi-ddsp.git
```

For dataset, please download the tfrecord files for the URMP dataset in
[here](https://console.cloud.google.com/storage/browser/magentadata/datasets/urmp/urmp_20210324) to the `data` folder in
your cloned repository using the following commands:

```
cd midi-ddsp # enter the project directory
mkdir ./data # create a data folder
gsutil cp gs://magentadata/datasets/urmp/urmp_20210324/* ./data # download tfrecords to directory
```

Please check [here](https://cloud.google.com/storage/docs/gsutil) for how to install and use `gsutil`.

Finally, you can run the script `train_midi_ddsp.sh` to train the exact same model we used in the paper:

```
sh ./train_midi_ddsp.sh
```

The current codebase does not support training with arbitrary dataset, but we will hopefully update that in the near
future.

Side note:

If one download the dataset to a different location, please change the `data_dir` parameter in `train_midi_ddsp.sh`.

The training of MIDI-DDSP takes approximately 18 hours on a single RTX 8000. The training code for now does not support
multi-GPU training. We recommend using a GPU with more than 24G of memory when training Synthesis Generator in batch
size of 16. For a GPU with less memory, please consider using a smaller batch size and change the batch size
in `train_midi_ddsp.sh`.

## Try to play with MIDI-DDSP yourself!

Please try out MIDI-DDSP
in [Colab notebooks](https://colab.research.google.com/github/magenta/midi-ddsp/blob/main/midi_ddsp/colab/MIDI_DDSP_Demo.ipynb)!

In this notebook, you will try to use MIDI-DDSP to synthesis a monophonic MIDI file, adjust note expressions, make pitch
bend by adjusting synthesis parameters, and synthesize quartet from Bach chorales.

We have trained MIDI-DDSP on the URMP dataset which support synthesizing 13 instruments: violin, viola, cello, double
bass, flute, oboe, clarinet, saxophone, bassoon, trumpet, horn, trombone, tuba. You could find how to download and use
our pre-trained model below:

## Command-line MIDI synthesis

On can use the MIDI-DDSP as a command-line MIDI synthesizer just like FluidSynth.

To use command-line synthesis to synthesize a midi file, please first download the model weights by running:

```
midi_ddsp_download_model_weights
```

To synthesize a midi file simply run the following command:

```
midi_ddsp_synthesize --midi_path <path-to-midi>
```

For a starter, you can try to synthesize the example midi file in this repository:

```
midi_ddsp_synthesize --midi_path ./midi_example/ode_to_joy.mid
```

The command line also enables synthesize a folder of midi files. For more advance use (synthesize a folder, using
FluidSynth for instruments not supported, etc.), please see `midi_ddsp_synthesize --help`.

If you have a trouble downloading the model weights, please manually download
from [here](https://github.com/magenta/midi-ddsp/raw/models/midi_ddsp_model_weights_urmp_9_10.zip), and specify
the `synthesis_generator_weight_path` and `expression_generator_weight_path` by yourself when using the command line.
You can also specify your other model weights if you want to use your own trained model.

## Python Usage

After installing midi-ddsp, you could import midi-ddsp in python and synthesize MIDI in your code.

### Minimal Example

Here is a simple example to use MIDI-DDSP to synthesize a midi file:

```python
from midi_ddsp import synthesize_midi, load_pretrained_model

midi_file = 'ode_to_joy.mid'
# Load pre-trained model
synthesis_generator, expression_generator = load_pretrained_model()
# Synthesize MIDI
output = synthesize_midi(synthesis_generator, expression_generator, midi_file)
# The synthesized audio
synthesized_audio = output['mix_audio']
```

### Advance Usage 

Here is an advance example to synthesize the `ode_to_joy.mid`, change the note expression controls, and adjust the
synthesis parameters:

```python
import numpy as np
import tensorflow as tf
from midi_ddsp.utils.midi_synthesis_utils import synthesize_mono_midi, conditioning_df_to_audio
from midi_ddsp.utils.inference_utils import get_process_group
from midi_ddsp.midi_ddsp_synthesize import load_pretrained_model
from midi_ddsp.data_handling.instrument_name_utils import INST_NAME_TO_ID_DICT

# -----MIDI Synthesis-----
midi_file = 'ode_to_joy.mid'
# Load pre-trained model
synthesis_generator, expression_generator = load_pretrained_model()
# Synthesize with violin:
instrument_name = 'violin'
instrument_id = INST_NAME_TO_ID_DICT[instrument_name]
# Run model prediction
midi_audio, midi_control_params, midi_synth_params, conditioning_df = synthesize_mono_midi(synthesis_generator,
                                                                                           expression_generator,
                                                                                           midi_file, instrument_id,
                                                                                           output_dir=None)

synthesized_audio = midi_audio  # The synthesized audio

# -----Adjust note expression controls and re-synthesize-----

# Make all notes weak vibrato:
conditioning_df_changed = conditioning_df.copy()
note_vibrato = conditioning_df_changed['vibrato'].value
conditioning_df_changed['vibrato'] = np.ones_like(conditioning_df['vibrato'].values) * 0.1
# Re-synthesize
midi_audio_changed, midi_control_params_changed, midi_synth_params_changed = conditioning_df_to_audio(
  synthesis_generator, conditioning_df_changed, tf.constant([instrument_id]))

synthesized_audio_changed = midi_audio_changed  # The synthesized audio

# There are 6 note expression controls in conditioning_df that you could change:
# 'volume', 'vol_fluc', 'vibrato', 'brightness', 'attack', 'vol_peak_pos'.
# Please refer to https://colab.research.google.com/github/magenta/midi-ddsp/blob/main/midi_ddsp/colab/MIDI_DDSP_Demo.ipynb#scrollTo=XfPPrdPu5sSy for the effect of each control. 

# -----Adjust synthesis parameters and re-synthesize-----

# The original synthesis parameters:
f0_ori = midi_synth_params['f0_hz']
amps_ori = midi_synth_params['amplitudes']
noise_ori = midi_synth_params['noise_magnitudes']
hd_ori = midi_synth_params['harmonic_distribution']

# TODO: make your change of the synthesis parameters here:
f0_changed = f0_ori
amps_changed = amps_ori
noise_changed = noise_ori
hd_changed = hd_ori

# Resynthesis the audio using DDSP
processor_group = get_process_group(midi_synth_params['amplitudes'].shape[1], use_angular_cumsum=True)
midi_audio_changed = processor_group({'amplitudes': amps_changed,
                                      'harmonic_distribution': hd_changed,
                                      'noise_magnitudes': noise_changed,
                                      'f0_hz': f0_changed, },
                                     verbose=False)
midi_audio_changed = synthesis_generator.reverb_module(midi_audio_changed, reverb_number=instrument_id, training=False)

synthesized_audio_changed = midi_audio_changed  # The synthesized audio
```
## Bibtex

```
@inproceedings{
wu2022mididdsp,
title={{MIDI}-{DDSP}: Detailed Control of Musical Performance via Hierarchical Modeling},
author={Yusong Wu and Ethan Manilow and Yi Deng and Rigel Swavely and Kyle Kastner and Tim Cooijmans and Aaron Courville and Cheng-Zhi Anna Huang and Jesse Engel},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=UseMOjWENv}
}
```

## Acknowledgment

**This is not an officially supported Google product.**

We would like to thank @akhaliq for creating the [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/midi-ddsp) (no longer functional).

[comment]: <> "## TODO:  0. Add more doc about python code synthesis api 1. Change the training loop, 2. Support multi-gpu training"
