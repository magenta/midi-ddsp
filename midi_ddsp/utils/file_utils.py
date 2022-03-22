"""Utility functions for file io and file path reading."""

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

import os
import shutil
import pickle
import json


def get_folder_name(path, num=1):
  """
  Get the name of the folder n levels above the given path.
  Example: a/b/c/d.txt, num=1 -> c, num=2 -> b, ...
  Args:
    path: a file path.
    num: the number of upper directories.

  Returns: the folder name for that level.

  """
  for _ in range(num):
    path = os.path.dirname(path)
  return os.path.basename(path)


def copy_file_to_folder(file_path, dst_dir):
  save_path = os.path.join(dst_dir, os.path.basename(file_path))
  shutil.copy(file_path, save_path)


def pickle_dump(obj, path):
  with open(path, 'wb') as f:
    pickle.dump(obj, f)
    f.close()


def pickle_load(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
    f.close()
  return data


def json_dump(data_json, json_save_path):
  with open(json_save_path, 'w') as f:
    json.dump(data_json, f)
    f.close()


def json_load(json_path):
  with open(json_path, 'r') as f:
    data = json.load(f)
    f.close()
  return data


def write_str_lines(save_path, lines):
  lines = [l + '\n' for l in lines]
  with open(save_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
