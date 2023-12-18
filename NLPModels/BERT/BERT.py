import tensorflow as tf
GPUconfig = tf.compat.v1.ConfigProto();
GPUconfig.gpu_options.per_process_gpu_memory_fraction = 0.4
from transformers import BertTokenizer, TFBertForSequenceClassification, BertConfig
import os
import numpy as np

class ModelController(object):
    def __init__(self, ModelOperationParameters, Tag):
        self.Tag = Tag
        self.OperationParameters = ModelOperationParameters
        self.Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        #self.Model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')
    def PrepareData(self, TrainDataFilename, MaxSequenceLength=512):
        input_ids = []
        attention_masks = []

        

        with open(TrainDataFilename, 'r', encoding='utf-8') as file:
            for line in file:
                tokens = self.Tokenizer.encode_plus(line, max_length=MaxSequenceLength, 
                                                    truncation=True, padding='max_length', 
                                                    add_special_tokens=True)
                input_ids.append(tokens['input_ids'])
                attention_masks.append(tokens['attention_mask'])


        return input_ids, attention_masks

    def Train(self, TrainData):
        pass

    def Validate(self, ValidData):
        pass

    def Test(self, TestData):
        pass

    def Run(self, DataMaker):
        CurrentPath = os.getcwd()
        DefaultDataOnly = self.OperationParameters.DefaultDataOnly
        DefaultDataPath = f'{CurrentPath}/{self.OperationParameters.DefaultDataPath}'

        if DefaultDataOnly is True:
            print(self.PrepareData(f"{DefaultDataPath}/train.txt"))
