import os;
import json;
import logging;
import DataMaker;
from RNN_LSTM_v2 import TrainAndValidLSTMModel,TestLSTMModel
AppName = os.getcwd().split("/")[-1];
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO);

class LSTMModelConfig(object):
    
    EpochPatience     = 3        # Number of epochs to wait before early stop if no progress on the validation set.
    MinDelta          = 0.001    # Minimum change in the monitored quantity to qualify as an improvement.
    MaxEpochNumber    = 50
    LearningRateDecay = 0.8
    VocabularySize    = 0
    ModelCheckPoint   = 'saves/model.ckpt'
    
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

class JsonDictToOBject:
    def __init__(self, Dict):
        for Key, Value in Dict.items():
            if isinstance(Value, dict):
                Value = JsonDictToOBject(Value)
            setattr(self, Key, Value)
    
if __name__ == "__main__":
    CurrentPath = os.getcwd();
    ModelConfig = LSTMModelConfig();
    logging.info(f"{AppName}: Reading configuration file..");
    with open("OperationParameters.json") as OperationParametersJsonFile:
        OperationParameters = JsonDictToOBject(json.load(OperationParametersJsonFile));
    DefaultDataOnly = OperationParameters.DefaultDataOnly;
        
    if OperationParameters.BasicStages.PrepareStage.StageEnabled is True and DefaultDataOnly is False:
        logging.info(f"{AppName}: PrepareStage is enabled. Running..");
        TrainSetsPath = OperationParameters.BasicStages.PrepareStage.RawTrainDataSetsPath;
        DataMaker.PrepareRawDataFiles(TrainSetsPath,
                                      f"{CurrentPath}/TrainData",
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileSemaCount,
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileContentsThreadsCount
        );
        TestSetsPath = OperationParameters.BasicStages.TestStage.TestSetsPath;
        DataMaker.PrepareRawDataFiles(TestSetsPath,
                                      f"{CurrentPath}/TestData",
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileSemaCount,
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileContentsThreadsCount
        );
    logging.info(f"{AppName}: Prepare all default(working) data sets");
    DefaultDataPath = f'{CurrentPath}/{OperationParameters.DefaultDataPath}';
    DefVocabularySize, DefDataSets = DataMaker.GetDataSetsDefault(DefaultDataPath=DefaultDataPath);
    ModelConfig.VocabularySize = DefVocabularySize; # vocabulary size preset
        
    if DefaultDataOnly is False:
        ModelConfig.ModelCheckPoint = 'saves/new_model.ckpt'
        VocabularySize, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData");
        ModelConfig.VocabularySize += VocabularySize;
    else:
        ModelConfig.ModelCheckPoint = 'saves/def_model.ckpt'
        
    if OperationParameters.BasicStages.TrainAndValidStage.StageEnabled is True:
        logging.info(f"{AppName}: TrainAndValidStage is enabled. Running..");
        if DefaultDataOnly is True:
            TrainAndValidLSTMModel(ModelConfig, {"Mode":"Default"}, DefDataSets["Train"], DefDataSets["Valid"]);
        else:
            TrainAndValidLSTMModel(ModelConfig,
                                   OperationParameters.BasicStages.BasicStagesScenario,
                                   DefDataSets["Train"]+GetReadySetFunc(f"{CurrentPath}/TrainData"), 
                                   DefDataSets["Valid"]);
            
    if OperationParameters.BasicStages.TestStage.StageEnabled is True:
        logging.info(f"{AppName}: TestStage is enabled. Running..");
        TestLSTMModel(ModelConfig, DefDataSets["Test"],"DefSet");
        if DefaultDataOnly is False:
            TestLSTMModel(ModelConfig, GetReadySetFunc(f"{CurrentPath}/TestData"),"NewSet");

