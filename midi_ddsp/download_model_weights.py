import zipfile
import os

try:
  from urllib.request import urlretrieve
except ImportError:
  from urllib import urlretrieve


def main():
  package_dir = os.path.dirname(os.path.realpath(__file__))
  weight_file_name = 'midi_ddsp_model_weights_urmp_9_10.zip'
  weight_file_download_path = os.path.join(package_dir, f'{weight_file_name}')
  print('Downloading midi-ddsp model weight files')
  urlretrieve(
    f'https://github.com/magenta/midi-ddsp/raw/models/{weight_file_name}',
    weight_file_download_path)
  print('Decompressing ...')
  with zipfile.ZipFile(weight_file_download_path, 'r') as zip_ref:
    zip_ref.extractall(package_dir)
  print('Decompression complete')
  os.remove(weight_file_download_path)


if __name__ == '__main__':
  main()
