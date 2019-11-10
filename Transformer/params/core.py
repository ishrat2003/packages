import argparse
import os
import logging
import json

class Core:

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_directory', default='/content/data/iwslt2016')
  parser.add_argument('--vocab_size', default=32000, type=int)

  def get(self):
    return self.parser.parse_args()

  def save(self, path):
    if not os.path.exists(path):
      os.makedirs(path)

    params = json.dumps(vars(self.get()))
    path = os.path.join(path, self._getFileName())
    with open(path, 'w') as fileToProcess:
        fileToProcess.write(params)

    logging.info("# Params saved in " + path)
    return

  def load(self, path):
    if not os.path.isdir(path):
        path = os.path.dirname(path)

    path = os.path.join(path, self._getFileName())
    fileContent = open(path, 'r').read()
    flag2val = json.loads(fileContent)
    for flag, value in flag2val.items():
        self.parser.flag = value

  def _getFileName(self):
    return type(self).__name__ + "params"