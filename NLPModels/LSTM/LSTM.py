import os;
import logging;
from .RNN_LSTM_v2 import TrainAndValidLSTMModel, TestLSTMModel

class ModelController(object):
    def __init__(self, ModelOperationParameters, Tag):
        self.Tag = Tag;
        self.OperationParameters = ModelOperationParameters;
        self.EpochPatience       = 3    # Number of epochs to wait before early stop if no progress on the validation set.
        self.MinDelta            = 0.001# Minimum change in the monitored quantity to qualify as an improvement.
        self.MaxEpochNumber      = 50
        self.LearningRateDecay   = 0.8
        self.VocabularySize      = 0
        self.ModelCheckPoint     = 'saves/LSTM.ckpt'
        self.BatchSize           = 20   # Determines how many data samples are processed in each forward/backward pass during training
        self.StepsPerEpoch       = 35   # Determines the fixed length of sequences in which data processed
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
    

    def Run(self, DataMaker):
        CurrentPath = os.getcwd();

        DefaultDataOnly = self.OperationParameters.DefaultDataOnly;
        DefaultDataPath = f'{CurrentPath}/{self.OperationParameters.DefaultDataPath}';

        if self.OperationParameters.BasicStages.PrepareStage.StageEnabled is True and DefaultDataOnly is False:
            logging.info(f"{self.Tag}: PrepareStage is enabled. Running..");
            TrainSetsPath = self.OperationParameters.BasicStages.PrepareStage.RawTrainDataSetsPath;
            DataMaker.PrepareRawDataFiles(TrainSetsPath,
                                          f"{CurrentPath}/TrainData",
                                          self.OperationParameters.BasicStages.PrepareStage.ProcessFileSemaCount,
                                          self.OperationParameters.BasicStages.PrepareStage.ProcessFileContentsThreadsCount,
            );
            DataMaker.MergeDataWithDefaultSet(DefaultDataPath, f"{CurrentPath}/TrainData");
            TestSetsPath = self.OperationParameters.BasicStages.TestStage.TestSetsPath;
            DataMaker.PrepareRawDataFiles(TestSetsPath,
                                          f"{CurrentPath}/TestData",
                                          self.OperationParameters.BasicStages.PrepareStage.ProcessFileSemaCount,
                                          self.OperationParameters.BasicStages.PrepareStage.ProcessFileContentsThreadsCount
            );
        logging.info(f"{self.Tag}: Prepare all default(working) data sets");

        DefVocabularySize, DefDataSets = DataMaker.GetDataSetsDefault(DefaultDataPath=DefaultDataPath);

        if self.OperationParameters.BasicStages.TrainAndValidStage.StageEnabled is True:
            logging.info(f"{self.Tag}: TrainAndValidStage is enabled. Running..");
            if DefaultDataOnly is True:
                self.VocabularySize = DefVocabularySize;
                self.RedefineParameters();
                TrainAndValidLSTMModel(self, DefDataSets["Train"], DefDataSets["Valid"]);
            else:
                if self.OperationParameters.BasicStages.BasicStagesScenario.Mode == "TrainDataAccumulator":
                        DataPart = self.OperationParameters.BasicStages.BasicStagesScenario.InitialDataPart;
                        VocabularyIncrease = self.OperationParameters.BasicStages.BasicStagesScenario.VocabularyIncrease;
                        InitialAdd = 0.0;
                        while DataPart <= 1.0:
                            CurrentTrainDataPart = DataPart+InitialAdd;
                            self.VocabularySize, IterationTrainData, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData", CurrentTrainDataPart);
                            self.RedefineParameters();
                            print(f'[DEBUG][TrainDataAccumulator]: CurrentTrainDataPart={CurrentTrainDataPart} VocabularySize={self.VocabularySize} ');
                            InitialAdd += VocabularyIncrease;
                            TrainAndValidLSTMModel(self, IterationTrainData, DefDataSets["Valid"]);
                            try:
                                os.rmdir("saves/") # TEMP
                            except:
                                pass;
                elif self.OperationParameters.BasicStages.BasicStagesScenario.Mode == "Default":
                    self.VocabularySize, TrainData, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData");
                    self.RedefineParameters();
                    TrainAndValidLSTMModel(self, TrainData, DefDataSets["Valid"]);


    #    if self.OperationParameters.BasicStages.TestStage.StageEnabled is True:
    #        logging.info(f"{AppName}: TestStage is enabled. Running..");
    #        TestLSTMModel(ModelConfig, DefDataSets["Test"],"DefSet");
    #        if DefaultDataOnly is False:
    #            TestLSTMModel(ModelConfig, GetReadySetFunc(f"{CurrentPath}/TestData"),"NewSet");