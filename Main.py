import os;
import json;
import logging;
import DataMaker;
from RNN_LSTM_v2 import TrainAndValidLSTMModel,TestLSTMModel
AppName = os.getcwd().split("/")[-1];
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO);

class LSTMModelConfig(object):
    
    EpochPatience     = 3        # Number of epochs to wait before early stop if no progress on the validation set.
    MinDelta          = 0.00001  # Minimum change in the monitored quantity to qualify as an improvement.
    MaxEpochNumber    = 50
    LearningRateDecay = 0.8
    VocabularySize    = 0
    
    # Global hyperparameters
    batch_size = 20
    max_grad_norm = 5
    learning_rate = 1.0
    init_scale = 0.05
    
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
    if DefaultDataOnly is False:
        VocabularySize, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData");
        ModelConfig.VocabularySize += VocabularySize;
    logging.info(f"{AppName}: Prepare all default(working) data sets");
    DefaultDataPath = f'{CurrentPath}/{OperationParameters["DefaultDataPath"]}';
    DefVocabularySize, DefDataSets = DataMaker.GetDataSetsDefault(DefaultDataPath=DefaultDataPath);
    ModelConfig.VocabularySize = DefVocabularySize; # vocabulary size preset
    
    if OperationParameters["PrepareStage"]["StageEnabled"] is True and DefaultDataOnly is False:
        logging.info(f"{AppName}: PrepareStage is enabled. Running..");
        DataMaker.PrepareRawDataFiles(OperationParameters["TrainSetsPath"],f"{CurrentPath}/TrainData");
        DataMaker.PrepareRawDataFiles(OperationParameters["TestSetsPath"],f"{CurrentPath}/TestData");
        
    if OperationParameters["TrainAndValidStage"]["StageEnabled"] is True:
        logging.info(f"{AppName}: TrainAndValidStage is enabled. Running..");
        if DefaultDataOnly is True:
            TrainAndValidLSTMModel(ModelConfig, DefDataSets["Train"], DefDataSets["Valid"]);
        else:
            TrainAndValidLSTMModel(ModelConfig, 
                                   DefDataSets["Train"]+GetReadySetFunc(f"{CurrentPath}/TrainData"), 
                                   DefDataSets["Valid"]);
            
    if OperationParameters["TestStage"]["StageEnabled"] is True:
        logging.info(f"{AppName}: TestStage is enabled. Running..");
        TestLSTMModel(ModelConfig, DefDataSets["Test"],"DefSet");
        if DefaultDataOnly is False:
            TestLSTMModel(ModelConfig, GetReadySetFunc(f"{CurrentPath}/TestData"),"NewSet");

