import os
import errno
import sentencepiece as spm
import logging

class Core:

  def __init__(self, params, dataSource):
    self.params = params
    self.dataSource = dataSource
    self.basePath = params.data_directory
    self.sp = None
    return
    
  def checkAllFilesExists(self):
    allFiles = self.dataSource.getFiles()
    if not allFiles:
      logging.error('Failed to identify data source files')
    
    for fileToCheck in allFiles:
        if not os.path.isfile(fileToCheck):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fileToCheck)

    return True

  def write(self, sentences, filePath):
    with open(filePath, 'w') as fileToProcess:
      fileToProcess.write("\n".join(sentences))
    return

  def loadTrainedBPE(self):
    self.sp = spm.SentencePieceProcessor()
    self.sp.Load(self.basePath + "/segmented/bpe.model")
    return

  def _segmentAndWrite(self, sentences, filePath):
    with open(filePath, "w") as fileToProcess:
      for sentence in sentences:
        pieces = self.sp.EncodeAsPieces(sentence)
        fileToProcess.write(" ".join(pieces) + "\n")

    

  