from .core import Core
import logging
import os
import sentencepiece as spm

class PreProcessing(Core):
  
  def execute(self):
    if not self.checkAllFilesExists():
      logging.error('Failed to execute pre-processing for the data source.')
      
    logging.info('# Processing data')
    self.dataSource.preProcessTrainingFiles()
    self.dataSource.preProcessEvaluationFiles()
    self.dataSource.preProcessTestingFiles()

    logging.info('# Printing sample data')
    self.dataSource.printSampleData()

    logging.info('# Write processed data')
    self.dataSource.writeAllProcessedFiles()
    return

  def trainBPE(self):
    logging.info("# Train a BPE model with sentencepiece")
    os.makedirs(self.params.data_directory + "/segmented", exist_ok=True)
    train = '--input=' + self.dataSource.getAllTextFile()
    train += ' --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 --model_prefix=' 
    train += str(self.params.data_directory)
    train += '/segmented/bpe --vocab_size=' + str(self.params.vocab_size) 
    train += ' --model_type=bpe'
    spm.SentencePieceTrainer.Train(train)
    return

  def writeBPE(self):
    self.dataSource.segmentAllData()
    self.dataSource.printSampleSegmentData()
    return





    