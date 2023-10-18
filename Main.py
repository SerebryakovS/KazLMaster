import os;
import json;
import logging;
import DataMaker;
from RNN_LSTM_v2 import TrainLSTMModel
AppName = "KazLMaster";
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO);


if __name__ == "__main__":
    with open("OperationParameters.json") as OperationParametersJsonFile:
        OperationParameters = json.load(OperationParametersJsonFile);
    logging.info(f"{AppName}: Reading configuration file..");
    # {
    #     "Prepare" : PrepareRawDataFiles(["Test":OperationParameters["TrainSetsPath"],"Valid"]);
    #     "Default" : LSTMFullProcessing(OperationParameters["DefaultDataPath"], OperationParameters["IsTestStepUsed"]),
    # }[OperationParameters["OperationMode"]];
    print(OperationParameters);

    CurrentPath = os.getcwd();
    # Data preparation flow:
        # DataMaker.PrepareRawDataFiles(OperationParameters["TrainSetsPath"],f"{CurrentPath}/TrainData");
        # DataMaker.PrepareRawDataFiles(OperationParameters["TestSetsPath"],f"{CurrentPath}/ValidData");


    VocabularySize, GetReadySetFunc = DataMaker.PrepareVocabulary(f"{CurrentPath}/TrainData");
    DataSets = [GetReadySetFunc(f"{CurrentPath}/TrainData"), GetReadySetFunc(f"{CurrentPath}/ValidData"),[]];
    TrainLSTMModel(VocabularySize, DataSets);
