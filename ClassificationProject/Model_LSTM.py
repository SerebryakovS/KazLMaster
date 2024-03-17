# Model_LSTM.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam

class LSTMModel:
    def __init__(self, MaxLength=128, VocabularySize=30522, EmbeddingDim=256, LSTMUnits=128, NumberOfClasses=4):
        self.Model = self.BuildModel(MaxLength, VocabularySize, EmbeddingDim, LSTMUnits, NumberOfClasses)
        print("[OK]: LSTM model building completed.")

    def BuildModel(self, MaxLength, VocabularySize, EmbeddingDim, LSTMUnits, NumberOfClasses):
        InputIds = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="input_ids")
        # Although attention_mask is not used, it's accepted for interface compatibility
        # AttentionMask = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="attention_mask")

        EmbeddingLayer = layers.Embedding(input_dim=VocabularySize, output_dim=EmbeddingDim, name='EmbeddingLayer')(InputIds)
        BiLSTMLayer = layers.Bidirectional(layers.LSTM(LSTMUnits, name='LSTMLayer'))(EmbeddingLayer)
        DenseLayer = layers.Dense(64, activation='relu')(BiLSTMLayer)
        OutputLayer = layers.Dense(NumberOfClasses, activation='sigmoid')(DenseLayer)

        Model = models.Model(inputs=InputIds, outputs=OutputLayer)
        Model.compile(optimizer=Adam(learning_rate=3e-5),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy', Precision(), Recall(), AUC()])
        return Model

    def Train(self, TrainDataset, ValidationDataset, EpochsCount=10):
        return self.Model.fit(TrainDataset, validation_data=ValidationDataset, epochs=EpochsCount)

    def Evaluate(self, TestDataset):
        return self.Model.evaluate(TestDataset)
