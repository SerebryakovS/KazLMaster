import os;
import json;
import logging;
import DataMaker;
from RNN_LSTM_v2 import TrainAndValidLSTMModel,TestLSTMModel
AppName = os.getcwd().split("/")[-1];
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO);

class LSTMModelConfig(object):
    # Global hyperparameters
    batch_size = 20
    max_grad_norm = 5
    lr_decay = 0.8
    learning_rate = 1.0
    init_scale = 0.05
    num_epochs = 20
    max_epoch = 6
    word_vocab_size = 0 # to be determined later
    num_layers = 2
    # RNN hyperparameters
    num_steps = 35
    hidden_size = 650
    keep_prob =0.5

if __name__ == "__main__":
    CurrentPath = os.getcwd();
    ModelConfig = LSTMModelConfig();
    logging.info(f"{AppName}: Reading configuration file..");
    with open("OperationParameters.json") as OperationParametersJsonFile:
        OperationParameters = json.load(OperationParametersJsonFile);
    DefaultDataOnly = OperationParameters["DefaultDataOnly"];
        
    logging.info(f"{AppName}: Prepare all default(working) data sets");
    DefaultDataPath = f'{CurrentPath}/{OperationParameters["DefaultDataPath"]}';
    DefVocabularySize, DefDataSets = DataMaker.GetDataSetsDefault(DefaultDataPath=DefaultDataPath);
    ModelConfig.word_vocab_size = DefVocabularySize; # vocabulary size preset
    
    if OperationParameters["PrepareStage"]["StageEnabled"] is True and DefaultDataOnly is False:
        logging.info(f"{AppName}: PrepareStage is enabled. Running..");

        
    if OperationParameters["TrainAndValidStage"]["StageEnabled"] is True:
        logging.info(f"{AppName}: TrainAndValidStage is enabled. Running..");
        if DefaultDataOnly is True:
            TrainAndValidLSTMModel(ModelConfig, DefDataSets["Train"], DefDataSets["Valid"]);
            
        else:
            pass; # for a while
            #DataMaker.PrepareRawDataFiles(OperationParameters["TrainSetsPath"],f"{CurrentPath}/TrainData");
            #DataMaker.PrepareRawDataFiles(OperationParameters["TestSetsPath"],f"{CurrentPath}/ValidData");
    
    if OperationParameters["TestStage"]["StageEnabled"] is True:
        logging.info(f"{AppName}: TestStage is enabled. Running..");
        if DefaultDataOnly is True:
            TestLSTMModel(ModelConfig, DefDataSets["Test"]);
            
        else:
            pass; # for a while


    #VocabularySize, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData");
    #DataSets = [GetReadySetFunc(f"{CurrentPath}/TrainData"), GetReadySetFunc(f"{CurrentPath}/ValidData"),[]];
    #TrainLSTMModel(VocabularySize, DataSets);
