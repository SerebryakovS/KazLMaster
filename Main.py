import os;
import json;
import logging;
import DataMaker;
from RNN_LSTM_v2 import TrainAndValidLSTMModel, TestLSTMModel
AppName = os.getcwd().split("/")[-1];
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO);

class LSTMModelConfig(object):
    def __init__(self):
        self.EpochPatience     = 3        # Number of epochs to wait before early stop if no progress on the validation set.
        self.MinDelta          = 0.001    # Minimum change in the monitored quantity to qualify as an improvement.
        self.MaxEpochNumber    = 50
        self.LearningRateDecay = 0.8
        self.VocabularySize    = 0
        self.ModelCheckPoint   = 'saves/model.ckpt'
        self.BatchSize         = 20       # Determines how many data samples are processed in each forward/backward pass during training
        self.StepsPerEpoch     = 35       # Determines the fixed length of sequences in which data processed
        # Global hyperparameters
        self.max_grad_norm = 5
        self.learning_rate = 1.0
        self.init_scale = 0.05
        self.num_layers = 2
        # RNN hyperparameters
        self.hidden_size = 650
        self.keep_prob =0.5
    def RedefineParameters(self):
        if self.VocabularySize <= 10000:
            self.BatchSize = 256;
            self.StepsPerEpoch = 100;
        elif self.VocabularySize <= 100000:
            self.BatchSize = 128;
            self.StepsPerEpoch = 60;
        elif self.VocabularySize <= 1000000:
            self.BatchSize = 64;
            self.StepsPerEpoch = 30;
        elif self.VocabularySize > 1000000:
            self.BatchSize = 16;
            self.StepsPerEpoch = 10;
            
class JsonDictToOBject:
    def __init__(self, Dict):
        for Key, Value in Dict.items():
            if isinstance(Value, dict):
                Value = JsonDictToOBject(Value)
            setattr(self, Key, Value)
    
if __name__ == "__main__":
    CurrentPath = os.getcwd();
    ModelConfig = LSTMModelConfig();
    ModelConfig.ModelCheckPoint = 'saves/model.ckpt'
    logging.info(f"{AppName}: Reading configuration file..");
    with open("OperationParameters.json") as OperationParametersJsonFile:
        OperationParameters = JsonDictToOBject(json.load(OperationParametersJsonFile));
    DefaultDataOnly = OperationParameters.DefaultDataOnly;
    DefaultDataPath = f'{CurrentPath}/{OperationParameters.DefaultDataPath}';
    
    if OperationParameters.BasicStages.PrepareStage.StageEnabled is True and DefaultDataOnly is False:
        logging.info(f"{AppName}: PrepareStage is enabled. Running..");
        TrainSetsPath = OperationParameters.BasicStages.PrepareStage.RawTrainDataSetsPath;
        DataMaker.PrepareRawDataFiles(TrainSetsPath,
                                      f"{CurrentPath}/TrainData",
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileSemaCount,
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileContentsThreadsCount,
        );
        DataMaker.MergeDataWithDefaultSet(DefaultDataPath, f"{CurrentPath}/TrainData");
        TestSetsPath = OperationParameters.BasicStages.TestStage.TestSetsPath;
        DataMaker.PrepareRawDataFiles(TestSetsPath,
                                      f"{CurrentPath}/TestData",
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileSemaCount,
                                      OperationParameters.BasicStages.PrepareStage.ProcessFileContentsThreadsCount
        );
    logging.info(f"{AppName}: Prepare all default(working) data sets");
    
    DefVocabularySize, DefDataSets = DataMaker.GetDataSetsDefault(DefaultDataPath=DefaultDataPath);
    

    if OperationParameters.BasicStages.TrainAndValidStage.StageEnabled is True:
        logging.info(f"{AppName}: TrainAndValidStage is enabled. Running..");
        if DefaultDataOnly is True:
            ModelConfig.VocabularySize = DefVocabularySize;
            ModelConfig.RedefineParameters();
            TrainAndValidLSTMModel(ModelConfig,  DefDataSets["Train"], DefDataSets["Valid"]);
        else:
            if OperationParameters.BasicStages.BasicStagesScenario.Mode == "TrainDataAccumulator":
                    DataPart = OperationParameters.BasicStages.BasicStagesScenario.InitialDataPart;
                    VocabularyIncrease = OperationParameters.BasicStages.BasicStagesScenario.VocabularyIncrease;
                    InitialAdd = 0.0;
                    while DataPart <= 1.0:
                        CurrentTrainDataPart = DataPart+InitialAdd;
                        ModelConfig.VocabularySize, IterationTrainData, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData", CurrentTrainDataPart);
                        ModelConfig.RedefineParameters();
                        print(f'[DEBUG][TrainDataAccumulator]: CurrentTrainDataPart={CurrentTrainDataPart} VocabularySize={ModelConfig.VocabularySize} ');
                        InitialAdd += VocabularyIncrease;
                        TrainAndValidLSTMModel(ModelConfig, IterationTrainData, DefDataSets["Valid"]);
                        try:
                            os.rmdir("saves/") # TEMP
                        except:
                            pass;
            elif OperationParameters.BasicStages.BasicStagesScenario.Mode == "Default":
                ModelConfig.VocabularySize, TrainData, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData");
                ModelConfig.RedefineParameters();
                TrainAndValidLSTMModel(ModelConfig, TrainData, DefDataSets["Valid"]);
   
            
#    if OperationParameters.BasicStages.TestStage.StageEnabled is True:
#        logging.info(f"{AppName}: TestStage is enabled. Running..");
#        TestLSTMModel(ModelConfig, DefDataSets["Test"],"DefSet");
#        if DefaultDataOnly is False:
#            TestLSTMModel(ModelConfig, GetReadySetFunc(f"{CurrentPath}/TestData"),"NewSet");

