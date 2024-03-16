# Model_BERT.py
import tensorflow as tf
from transformers import TFBertForSequenceClassification
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

class BERTModel:
    ModelName = 'bert-base-multilingual-cased';
    def __init__(self, MaxLength=128):
        self.Model = self.BuildModel(MaxLength);
        print("[OK]: BERT model building completed.");
    def BuildModel(self, MaxLength):
        BertLayer = TFBertForSequenceClassification.from_pretrained(BERTModel.ModelName, num_labels=4);
        BertLayer.trainable = True;
        InputIds = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="input_ids");
        AttentionMask = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="attention_mask");
        BertOutput = BertLayer(InputIds, attention_mask=AttentionMask).logits;
        Model = models.Model(inputs=[InputIds, AttentionMask], outputs=BertOutput);
        Model.compile(optimizer=Adam(learning_rate=3e-5),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy', Precision(), Recall(), AUC()]);
        return Model;

    def Train(self, TrainDataset, ValidationDataset, EpochsCount=10):
        return self.Model.fit(TrainDataset, validation_data=ValidationDataset, epochs=EpochsCount)

    def Evaluate(self, TestDataset):
        return self.Model.evaluate(TestDataset)
