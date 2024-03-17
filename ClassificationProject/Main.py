# Main.py
from Model_LSTM import LSTMModel
from Model_BERT import BERTModel
from Model_GPT import GPTModel
from Data import DataPreparation
import matplotlib.pyplot as Plot
import numpy as np

def PlotMetric(HistoryObjects, Labels, BaseMetric, MetricType):
    Plot.figure(figsize=(12, 8))
    for History, Label in zip(HistoryObjects, Labels):
        MetricName = f'{MetricType}_{BaseMetric}' if f'{MetricType}_{BaseMetric}' in History.history else BaseMetric
        if MetricType == 'val':
            MetricName = 'val_' + MetricName

        if MetricName in History.history:
            MetricValues = History.history[MetricName]
            Epochs = range(1, len(MetricValues) + 1)
            Plot.plot(Epochs, MetricValues, 'o-', label=f'{Label} {MetricType} {BaseMetric.capitalize()}')

    Plot.title(f'{MetricType.capitalize()} {BaseMetric.capitalize()}')
    Plot.xlabel('Epochs')
    Plot.ylabel(BaseMetric.capitalize())
    Plot.legend()
    Plot.savefig(f'{MetricType}_{BaseMetric}.jpg')
    Plot.close()

def PlotMetrics(HistoryObjects, Labels):
    BaseMetrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']

    # Adjust metric names for BERT and GPT if needed
    AdjustedHistories = []
    for History in HistoryObjects:
        AdjustedHistory = {}
        for key, value in History.history.items():
            new_key = key.replace(' 1', '').replace(' 2', '')
            AdjustedHistory[new_key] = value
        AdjustedHistories.append(type('History', (object,), {'history': AdjustedHistory}))

    for BaseMetric in BaseMetrics:
        # Plot training metrics
        PlotMetric(AdjustedHistories, Labels, BaseMetric, 'train')
        # Plot validation metrics
        PlotMetric(AdjustedHistories, Labels, BaseMetric, 'val')


def PrintEvaluationResults(Results, HistoryObjects, Labels):
    print("Model Evaluation Results:")
    for Result, History, label in zip(Results, HistoryObjects, Labels):
        print(f"\n{label} Results:");
        MetricsNames = list(History.history.keys());
        for Idx, MetricName in enumerate(MetricsNames):
            if 'val_' in MetricName:
                MetricValue = History.history[MetricName][-1];
            else:
                MetricValue = Result[Idx];
            print(f"\t{MetricName.replace('_', ' ').capitalize()}: {MetricValue:.4f}");

def Main():
    CountEpochs = 10;
    DataPrepare = DataPreparation(
        DatasetName="yeshpanovrustem/ner-kazakh", BertModelName=BERTModel.ModelName, GptModelName=GPTModel.ModelName
    );
    #######################################################################################################################
    TrainSet, ValidSet, TestSet = DataPrepare.GetDatasets(ModelType="LSTM");
    LSTM = LSTMModel( MaxLength       = 128,
                      VocabularySize  = DataPrepare.VocabularySize,
                      EmbeddingDim    = 256, # Dimensionality of the embedding layer. Common values are 50, 100, 256, and 512.
                      LSTMUnits       = 128, # Represents the number of LSTM units (or neurons) in the LSTM layer
                      NumberOfClasses = 4
    );
    LSTMResults = [LSTM.Train(TrainSet, ValidSet, CountEpochs), LSTM.Evaluate(TestSet)];
    #######################################################################################################################
    TrainSet, ValidSet, TestSet = DataPrepare.GetDatasets(ModelType="BERT");
    BERT = BERTModel(MaxLength=128);
    BERTResults = [BERT.Train(TrainSet, ValidSet, CountEpochs), BERT.Evaluate(TestSet)];
    #######################################################################################################################
    TrainSet, ValidSet, TestSet = DataPrepare.GetDatasets(ModelType="GPT");
    GPT = GPTModel(MaxLength=128);
    GPTResults = [GPT.Train(TrainSet, ValidSet, CountEpochs), GPT.Evaluate(TestSet)];
    #######################################################################################################################
    HistoryObjects = [LSTMResults[0], BERTResults[0], GPTResults[0]];
    Results        = [LSTMResults[1], BERTResults[1], GPTResults[1]];
    Labels         = ['LSTM', 'BERT', 'GPT'];

    # HistoryObjects = [LSTMResults[0]];
    # Results        = [LSTMResults[1]];
    # Labels         = ['LSTM'];

    PlotMetrics(HistoryObjects, Labels);
    PrintEvaluationResults(Results, HistoryObjects, Labels);

if __name__ == "__main__":
    Main();










