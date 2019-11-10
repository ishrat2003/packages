import logging
from model import Transformer
from params.preprocessing import PreProcessing as PreProcessingParams
from data_processor.preprocessing import PreProcessing as PreProcessingProcessor
from data_processor.iwslt import IWSLT

logging.basicConfig(level=logging.INFO)

logging.info("# Loading script params ")
paramsProcessor = PreProcessingParams()
params = paramsProcessor.get()
paramsProcessor.save(params.data_directory)

dataSource = IWSLT(params.data_directory)

processor = PreProcessingProcessor(params, dataSource)
processor.execute()
processor.trainBPE()
processor.writeBPE()
logging.info('Finished')