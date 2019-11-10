from .core import Core
import logging
import re
import os

class IWSLT(Core):

  def __init__(self, basePath, params):
    self.params = params
    self.basePath = basePath
    self.trainFiles = [
      basePath + "/de-en/train.tags.de-en.de",
      basePath + "/de-en/train.tags.de-en.en"
    ]
    self.evaluateFiles = [
      basePath + "/de-en/IWSLT16.TED.tst2013.de-en.de.xml",
      basePath + "/de-en/IWSLT16.TED.tst2013.de-en.en.xml"
    ]
    self.testFiles = [
      basePath + "/de-en/IWSLT16.TED.tst2014.de-en.de.xml",
      basePath + "/de-en/IWSLT16.TED.tst2014.de-en.en.xml"
    ]
    self.trainData = {}
    self.evaluateData = {}
    self.testData = {}
    
    return

  def getFiles(self):
    return self.trainFiles + self.evaluateFiles + self.testFiles

  def getAllTextFile(self):
    return self.basePath + "/prepro/train"

  def get_batch(fpath1, fpath2, maxlen1, maxlen2, vocab_fpath, batch_size, shuffle=False):
    '''Gets training / evaluation mini-batches
    fpath1: source file path. string.
    fpath2: target file path. string.
    maxlen1: source sent maximum length. scalar.
    maxlen2: target sent maximum length. scalar.
    vocab_fpath: string. vocabulary file path.
    batch_size: scalar
    shuffle: boolean
    Returns
    batches
    num_batches: number of mini-batches
    num_samples
    '''


    sents1, sents2 = load_data(fpath1, fpath2, maxlen1, maxlen2)
    batches = input_fn(sents1, sents2, vocab_fpath, batch_size, shuffle=shuffle)
    num_batches = calc_num_batches(len(sents1), batch_size)
    return batches, num_batches, len(sents1)

  def writeAllProcessedFiles(self):
    basePath = self.basePath + "/prepro"
    os.makedirs(basePath, exist_ok=True)
    self.write(self.trainData['sequence1'], basePath + "/train.de")
    self.write(self.trainData['sequence2'], basePath + "/train.en")
    self.write(self.trainData['sequence1'] + self.trainData['sequence2'], basePath + "/train")
    self.write(self.evaluateData['sequence1'], basePath + "/eval.de")
    self.write(self.evaluateData['sequence2'], basePath + "/eval.en")
    self.write(self.testData['sequence1'], basePath + "/test.de")
    self.write(self.testData['sequence2'], basePath + "/test.en")
    return

  def segmentAllData(self):
    basePath = self.basePath + "/segmented"
    os.makedirs(basePath, exist_ok=True)
    self.loadTrainedBPE()
    self._segmentAndWrite(self.trainData['sequence1'], basePath + "/train.de.bpe")
    self._segmentAndWrite(self.trainData['sequence2'], basePath + "/train.en.bpe")
    self._segmentAndWrite(self.evaluateData['sequence1'], basePath + "/eval.de.bpe")
    self._segmentAndWrite(self.evaluateData['sequence2'], basePath + "/eval.en.bpe")
    self._segmentAndWrite(self.testData['sequence1'], basePath + "/test.de.bpe")
    self._segmentAndWrite(self.testData['sequence2'], basePath + "/test.en.bpe")
    return

  def printSampleData(self):
    logging.info("Train seq1: " + str(self.trainData['sequence1'][0]))
    logging.info("Train seq2: " + str(self.trainData['sequence2'][0]))
    logging.info("Eval seq1: " + str(self.evaluateData['sequence1'][0]))
    logging.info("Eval seq2: " + str(self.evaluateData['sequence2'][0]))
    logging.info("Test seq1: " + str(self.testData['sequence1'][0]))
    logging.info("Test seq2: " + str(self.testData['sequence2'][0]))
    return

  def printSampleSegmentData(self):
    basePath = self.basePath + "/segmented"
    logging.info("Let's see how segmented data look like")
    print("train1:", open(basePath + "/train.de.bpe",'r').readline())
    print("train2:", open(basePath + "/train.en.bpe", 'r').readline())
    print("eval1:", open(basePath + "/eval.de.bpe", 'r').readline())
    print("eval2:", open(basePath + "/eval.en.bpe", 'r').readline())
    print("test1:", open(basePath + "/test.de.bpe", 'r').readline())
    print("test1:", open(basePath + "/test.en.bpe", 'r').readline())
    return

  def preProcessTrainingFiles(self):
    self.trainData['sequence1'] = self._convert(self.trainFiles[0])
    self.trainData['sequence2'] = self._convert(self.trainFiles[1])
    assert len(self.trainData['sequence1'])==len(self.trainData['sequence2'])
    return

  def preProcessEvaluationFiles(self):
    self.evaluateData['sequence1'] = self._convertXml(self.evaluateFiles[0])
    self.evaluateData['sequence2'] = self._convertXml(self.evaluateFiles[1])
    assert len(self.evaluateData['sequence1'])==len(self.evaluateData['sequence2'])
    return

  def preProcessTestingFiles(self):
    self.testData['sequence1'] = self._convertXml(self.testFiles[0])
    self.testData['sequence2'] = self._convertXml(self.testFiles[1])
    assert len(self.testData['sequence1'])==len(self.testData['sequence2'])
    return

  def _convert(self, filePath):
    return [line.strip() \
      for line in open(filePath, 'r').read().split("\n") \
      if not line.startswith("<")]

  def _convertXml(self, filePath):
    return [re.sub("<[^>]+>", "", line).strip() \
      for line in open(filePath, 'r').read().split("\n") \
      if line.startswith("<seg id")]