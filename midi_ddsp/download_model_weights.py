import zipfile
import os

try:
  from urllib.request import urlretrieve
except ImportError:
  from urllib import urlretrieve


def main():
  package_dir = os.path.dirname(os.path.realpath(__file__))
  weight_file_name = 'midi_ddsp_model_weights_urmp_9_10.zip'
  print('Downloading midi-ddsp model weight files')
  urlretrieve(
    f'https://github.com/lukewys/midi-ddsp/raw/models/{weight_file_name}',
    os.path.join(package_dir, f'{weight_file_name}'))
  print('Decompressing ...')
  with zipfile.ZipFile(f'{weight_file_name}', 'r') as zip_ref:
    zip_ref.extractall('./')
  print('Decompression complete')


if __name__ == '__main__':
  main()
