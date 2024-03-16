# Main.py
from Model_LSTM import LSTMModel
from Model_BERT import BERTModel
from Model_GPT import GPTModel
from Data import DataPreparation
import matplotlib.pyplot as Plot
import numpy as np

def PlotMetrics(Histories, Labels):
    BaseMetrics = ['loss', 'accuracy', 'precision', 'recall', 'auc'];
    for BaseMetric in BaseMetrics:
        SpecificMetrics = [f'{task}_{BaseMetric}' for task in ['time_sensitive', 'organizational_interaction', 'personal_relevance', 'commercial_intent']];
        Plot.figure(figsize=(12, 8));
        for History, Label in zip(Histories, Labels):
            print("_______________",History.history);
            for SpecificMetric in SpecificMetrics:
                if SpecificMetric in History.history:
                    TrainMetric = History.history[SpecificMetric];
                    ValidMetric = History.history[f'val_{SpecificMetric}'];
                    Epochs = range(1, len(TrainMetric) + 1);
                    Plot.plot(Epochs, TrainMetric, 'o-', label=f'{Label} {SpecificMetric.capitalize()} Training');
                    Plot.plot(Epochs, ValidMetric, 'o--', label=f'{Label} {SpecificMetric.capitalize()} Validation');
        Plot.title(f'Training and Validation {BaseMetric.capitalize()}');
        Plot.xlabel('Epochs');
        Plot.ylabel(BaseMetric.capitalize());
        Plot.legend();
        Plot.savefig(f'ModelComparison_{BaseMetric}.jpg');
        Plot.close();

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
    CountEpochs = 1;
    DataPrepare = DataPreparation(
        DatasetName="yeshpanovrustem/ner-kazakh", BertModelName=BERTModel.ModelName, GptModelName=GPTModel.ModelName
    );
    #######################################################################################################################
    TrainSet, ValidSet, TestSet = DataPrepare.GetDatasets(ModelType="LSTM");
    LSTM = LSTMModel(DataPrepare.VocabularySize,
                     EmbeddingDim = 256, # Dimensionality of the embedding layer. Common values are 50, 100, 256, and 512.
                     LSTMUnits    = 128  # Represents the number of LSTM units (or neurons) in the LSTM layer
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

    # HistoryObjects = [BERTResults[0]];
    # Results        = [BERTResults[1]];
    # Labels         = ['BERT'];


    PlotMetrics(HistoryObjects, Labels);
    PrintEvaluationResults(Results, HistoryObjects, Labels);

if __name__ == "__main__":
    Main();










