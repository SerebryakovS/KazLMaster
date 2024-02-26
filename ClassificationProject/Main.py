# Main.py
from Model_LSTM import LSTMModel
from Model_BERT import BERTModel
from Data import DataPreparation
import matplotlib.pyplot as Plot
import numpy as np

def PlotHistory(History, ModelName):
    TrainAccuracyKeys = [Key for Key in History.history.keys() if "accuracy" in Key and "val" not in Key];
    ValueAccuracyKeys = [Key for Key in History.history.keys() if "val_accuracy" in Key];
    AverageTrainAccuracy = np.mean([History.history[Key] for Key in TrainAccuracyKeys], axis=0);
    AverageValueAccuracy = np.mean([History.history[Key] for Key in ValueAccuracyKeys], axis=0);
    # Plot 1: Training and Validation Loss
    Plot.figure(figsize=(10, 6));
    Plot.plot(History.history['loss'], label='Training Loss');
    Plot.plot(History.history['val_loss'], label='Validation Loss');
    Plot.title(f'{ModelName} Training and Validation Loss');
    Plot.xlabel('Epochs');
    Plot.ylabel('Loss');
    Plot.legend();
    Plot.savefig(f'{ModelName}_Loss.jpg');
    Plot.close();
    # Plot 2: Average Training and Validation Accuracy
    Plot.figure(figsize=(10, 6));
    Plot.plot(AverageTrainAccuracy, label='Average Training Accuracy');
    Plot.plot(AverageValueAccuracy, label='Average Validation Accuracy');
    Plot.title(f'{ModelName} Average Training and Validation Accuracy');
    Plot.xlabel('Epochs');
    Plot.ylabel('Accuracy');
    Plot.legend();
    Plot.savefig(f'{ModelName}_Accuracy.jpg');
    Plot.close();

def Main():
    CountEpochs = 20;
    DataPrepare = DataPreparation();
    ######################################################################################################################
    TrainSet, ValidSet, TestSet = DataPrepare.GetDatasets(ModelType="LSTM");
    LSTM = LSTMModel(DataPrepare.VocabularySize,
                     EmbeddingDim = 256, # Dimensionality of the embedding layer. Common values are 50, 100, 256, and 512.
                     LSTMUnits    = 128  # Represents the number of LSTM units (or neurons) in the LSTM layer
    );
    PlotHistory(LSTM.Train(TrainSet, ValidSet, CountEpochs), "LSTM");
    print(LSTM.Evaluate(TestSet));
    ######################################################################################################################
    TrainSet, ValidSet, TestSet = DataPrepare.GetDatasets(ModelType="BERT");
    BERT = BERTModel();
    PlotHistory(BERT.Train(TrainSet, ValidSet, CountEpochs), "BERT");
    print(BERT.Evaluate(TestSet));
    ######################################################################################################################
if __name__ == "__main__":
    Main();
