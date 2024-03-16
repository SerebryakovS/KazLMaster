# Model_LSTM.py
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras import layers, models

class LSTMModel:
    def __init__(self, VocabularySize, EmbeddingDim=256, LSTMUnits=128):
        self.Model = self.BuildModel(VocabularySize, EmbeddingDim, LSTMUnits)
        print("[OK]: LSTM model building completed.")

    def BuildModel(self, VocabularySize, EmbeddingDim, LSTMUnits):
        InputLayer = layers.Input(shape=(None,), dtype='int32', name='InputLayer')
        EmbeddingLayer = layers.Embedding(VocabularySize, EmbeddingDim, name='EmbeddingLayer')(InputLayer)
        BiLSTMLayer = layers.Bidirectional(layers.LSTM(LSTMUnits, name='LSTMLayer'), name='BiLSTMLayer')(EmbeddingLayer)
        DenseLayer = layers.Dense(64, activation='relu', name='DenseLayer')(BiLSTMLayer)

        TimeSensitiveOutput = layers.Dense(1, activation='sigmoid', name='time_sensitive')(DenseLayer)
        OrganizationalInteractionOutput = layers.Dense(1, activation='sigmoid', name='organizational_interaction')(DenseLayer)
        PersonalRelevanceOutput = layers.Dense(1, activation='sigmoid', name='personal_relevance')(DenseLayer)
        CommercialIntentOutput = layers.Dense(1, activation='sigmoid', name='commercial_intent')(DenseLayer)

        Model = models.Model(inputs=InputLayer, outputs=[
            TimeSensitiveOutput,
            OrganizationalInteractionOutput,
            PersonalRelevanceOutput,
            CommercialIntentOutput
        ])

        Model.compile(optimizer='adam',
                      loss={
                          'time_sensitive': 'binary_crossentropy',
                          'organizational_interaction': 'binary_crossentropy',
                          'personal_relevance': 'binary_crossentropy',
                          'commercial_intent': 'binary_crossentropy'
                      },
                      metrics=['accuracy', Precision(), Recall(), AUC()]);
        return Model

    def Train(self, TrainDataset, ValidationDataset, EpochsCount=10):
        return self.Model.fit(TrainDataset, validation_data=ValidationDataset, epochs=EpochsCount)

    def Evaluate(self, TestDataset):
        return self.Model.evaluate(TestDataset)
