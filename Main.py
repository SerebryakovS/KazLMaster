import os;
import json;
import logging;
import DataMaker
from NLPModels.LSTM import LSTM
# from NLPModels.BERT import BERT

AppName = os.getcwd().split("/")[-1];
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO);

            
class JsonDictToOBject:
    def __init__(self, Dict):
        for Key, Value in Dict.items():
            if isinstance(Value, dict):
                Value = JsonDictToOBject(Value)
            setattr(self, Key, Value)

def ReadJsonConfig(ConfigFilePath):
    logging.info(f"{AppName}: Reading configuration file..");
    with open(ConfigFilePath) as OperationParametersJsonFile:
        return JsonDictToOBject(json.load(OperationParametersJsonFile));
    
if __name__ == "__main__":
    MainConfig = ReadJsonConfig(f"MainConfig.json");
    logging.info(f"{AppName}: Selected model is: {MainConfig.NLPModel}");
    ModelOperationParameters = ReadJsonConfig(f"NLPModels/{MainConfig.NLPModel}/OperationParameters.json");
    
    ModelController = {
        "LSTM" : LSTM.ModelController(ModelOperationParameters, AppName),
        # "BERT" : BERT.ModelController(ModelOperationParameters, AppName)
    }[MainConfig.NLPModel]
        
    ModelController.Run(DataMaker);
    
