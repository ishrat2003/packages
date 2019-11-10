import tensorflow as tf
import logging
from model import Transformer
from params.train import Train as TrainParams

logging.basicConfig(level=logging.INFO)

logging.info("# Loading script params ")
trainingParams = TrainParams()
params = trainingParams.get()
trainingParams.save(params.data_directory)

logging.info("# Prepare train/eval batches")

logging.info('Finished')