# Model_GPT.py
import tensorflow as tf
from transformers import TFGPT2ForSequenceClassification, GPT2Tokenizer
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

class GPTModel:
    ModelName = 'gpt2';
    def __init__(self, MaxLength=128):
        self.Model = self.BuildModel(MaxLength);
        print("[OK]: GPT model building completed.");

    def BuildModel(self, MaxLength):
        GptLayer = TFGPT2ForSequenceClassification.from_pretrained(GPTModel.ModelName, num_labels=4);
        InputIds = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="input_ids");
        AttentionMask = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="attention_mask");

        GptOutputs = GptLayer(InputIds, attention_mask=AttentionMask);
        Logits = GptOutputs.logits;
        Logits = tf.reduce_mean(Logits, axis=1);

        Model = models.Model(inputs=[InputIds, AttentionMask], outputs=Logits)
        Model.compile(optimizer=Adam(learning_rate=3e-5),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy', Precision(), Recall(), AUC()]);
        return Model;

    def Train(self, TrainDataset, ValidationDataset, EpochsCount=10):
        return self.Model.fit(TrainDataset, validation_data=ValidationDataset, epochs=EpochsCount);

    def Evaluate(self, TestDataset):
        return self.Model.evaluate(TestDataset);
