<div align="center">
<img src="https://midi-ddsp.github.io/pics/midi-ddsp-logo.png" width="200px" alt="logo"></img>
</div>

# MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling

[Demos](https://midi-ddsp.github.io/) | Blog Post
| [Colab Notebook](https://colab.research.google.com/github/magenta/midi-ddsp/blob/main/midi_ddsp/colab/MIDI_DDSP_Demo.ipynb)
| Paper

MIDI-DDSP is a hierarchical audio generation model for synthesizing MIDI expanded
from [DDSP](https://github.com/magenta/ddsp).

## Links

* [Check out the blog post 💻]()
* [Read the original paper 📄]()
* [Listen to some examples 🔈](https://midi-ddsp.github.io/)
* [Try out MIDI Synthesis using MIDI-DDSP 🎵->🎻🔊](https://colab.research.google.com/github/magenta/midi-ddsp/blob/main/midi_ddsp/colab/MIDI_DDSP_Demo.ipynb)

## Install MIDI-DDSP

You could install MIDI-DDSP via pip, which allows you to use the
cool [Command-line MIDI synthesis](#command-line-midi-synthesis) to synthesize your MIDI.

To install MIDI-DDSP via pip, simply run:

```
pip install midi-ddsp
```

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

## Command-line MIDI synthesis

On can use the MIDI-DDSP as a command-line MIDI synthesizer just like FluidSynth.

To use command-line synthesis to synthesize a midi file, please first download the model weights by running:

```
midi_ddsp_download_model_weights
```

To synthesize a midi file simply run the following command:

```
python midi_ddsp_synthesize.py --midi_path <path-to-midi>
```

For a starter, you can try to synthesize the example midi file in this repository:

```
python midi_ddsp_synthesize.py --midi_path ./midi_example/ode_to_joy.mid
```

The command line also enables synthesize a folder of midi files. For more advance use (synthesize a folder, using
FluidSynth for instruments not supported, etc.), please see `synthesize_midi.py --help`.

If you have a trouble downloading the model weights, please manually download
from [here](https://github.com/magenta/midi-ddsp/raw/models/midi_ddsp_model_weights_urmp_9_10.zip), and specify
the `synthesis_generator_weight_path` and `expression_generator_weight_path` by yourself when using the command line.
You can also specify your other model weights if you want to use your own trained model

[comment]: <> (## TODO: 0. Add script, dealing with model weight download, 1. Change the training loop, 2. Support multi-gpu training)