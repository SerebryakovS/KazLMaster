# Model_BERT.py
import tensorflow as tf
from transformers import TFBertForSequenceClassification
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

class BERTModel:
    ModelName = 'bert-base-multilingual-cased';
    def __init__(self, MaxLength=128):
        self.Model = self.BuildModel(MaxLength);
        print("[OK]: BERT model building completed.");
    def BuildModel(self, MaxLength):
        BertLayer = TFBertForSequenceClassification.from_pretrained(BERTModel.ModelName, num_labels=4);
        BertLayer.trainable = False
        InputIds = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="input_ids")
        AttentionMask = layers.Input(shape=(MaxLength,), dtype=tf.int32, name="attention_mask")
        BertInputs = {'input_ids': InputIds, 'attention_mask': AttentionMask}
        BertOutput = BertLayer(BertInputs)[0]
        DenseLayer = layers.Dense(64, activation='relu', name='DenseLayer')(BertOutput)

        TimeSensitiveOutput = layers.Dense(1, activation='sigmoid', name='time_sensitive')(DenseLayer)
        OrganizationalInteractionOutput = layers.Dense(1, activation='sigmoid', name='organizational_interaction')(DenseLayer)
        PersonalRelevanceOutput = layers.Dense(1, activation='sigmoid', name='personal_relevance')(DenseLayer)
        CommercialIntentOutput = layers.Dense(1, activation='sigmoid', name='commercial_intent')(DenseLayer)

        Model = models.Model(inputs=[InputIds, AttentionMask], outputs=[
            TimeSensitiveOutput,
            OrganizationalInteractionOutput,
            PersonalRelevanceOutput,
            CommercialIntentOutput
        ])

        Model.compile(optimizer=Adam(learning_rate=3e-5),
                      loss={
                          'time_sensitive': 'binary_crossentropy',
                          'organizational_interaction': 'binary_crossentropy',
                          'personal_relevance': 'binary_crossentropy',
                          'commercial_intent': 'binary_crossentropy'
                      },
                      metrics=['accuracy'])
        return Model

    def Train(self, TrainDataset, ValidationDataset, EpochsCount=10):
        return self.Model.fit(TrainDataset, validation_data=ValidationDataset, epochs=EpochsCount)

    def Evaluate(self, TestDataset):
        return self.Model.evaluate(TestDataset)
