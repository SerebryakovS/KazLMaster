# Data.py
from datasets import load_dataset
import tensorflow as tf
from transformers import BertTokenizer, GPT2Tokenizer
import numpy as np

def DetermineTimeSensitiveLabel(SimplifiedTags):
    TimeSensitiveEntities = {'DATE', 'TIME', 'EVENT'};
    return 1 if any(Tag in TimeSensitiveEntities for Tag in SimplifiedTags) else 0

def DetermineOrganizationalInteractionLabel(SimplifiedTags):
    OrganizationalEntities = {'ORGANISATION', 'CONTACT', 'LAW', 'MONEY', 'FACILITY'};
    return 1 if any(Tag in OrganizationalEntities for Tag in SimplifiedTags) else 0

def DeterminePersonalRelevanceLabel(SimplifiedTags):
    PersonalRelevantEntities = {'PERSON', 'POSITION', 'CONTACT'};
    return 1 if any(Tag in PersonalRelevantEntities for Tag in SimplifiedTags) else 0

def DetermineCommercialIntentLabel(SimplifiedTags):
    CommercialEntities = {'MONEY', 'PRODUCT', 'ORGANISATION', 'CONTACT'};
    return 1 if any(Tag in CommercialEntities for Tag in SimplifiedTags) else 0

class DataPreparation:
    def __init__(self, DatasetName, BertModelName, GptModelName):
        self.HuggingFaceDataset = load_dataset(DatasetName);
        print(f"[OK]: Dataset loaded: {self.HuggingFaceDataset}");
        self.CalculateVocabulary();
        self.AddClassificationFlags();
        self.BertTokenizer = BertTokenizer.from_pretrained(BertModelName);
        self.GptTokenizer  = GPT2Tokenizer.from_pretrained(GptModelName);
    def CalculateVocabulary(self):
        AllTokens = [Token for Sample in self.HuggingFaceDataset['train'] for Token in Sample['tokens']];
        self.Token2Index = {Token: Idx for Idx, Token in enumerate(sorted(set(AllTokens)))};
        self.Index2Token = {Idx: Token for Token, Idx in self.Token2Index.items()};
        self.VocabularySize = len(self.Token2Index);
        print(f"[OK]: VocabularySize calculated: {self.VocabularySize}");
    def PreprocessNerTags(self, NERTagsSequence):
        OriginalNERTagsNames = [
            "O",
            "B-ADAGE",
            "I-ADAGE",
            "B-ART",
            "I-ART",
            "B-CARDINAL",
            "I-CARDINAL",
            "B-CONTACT",
            "I-CONTACT",
            "B-DATE",
            "I-DATE",
            "B-DISEASE",
            "I-DISEASE",
            "B-EVENT",
            "I-EVENT",
            "B-FACILITY",
            "I-FACILITY",
            "B-GPE",
            "I-GPE",
            "B-LANGUAGE",
            "I-LANGUAGE",
            "B-LAW",
            "I-LAW",
            "B-LOCATION",
            "I-LOCATION",
            "B-MISCELLANEOUS",
            "I-MISCELLANEOUS",
            "B-MONEY",
            "I-MONEY",
            "B-NON_HUMAN",
            "I-NON_HUMAN",
            "B-NORP",
            "I-NORP",
            "B-ORDINAL",
            "I-ORDINAL",
            "B-ORGANISATION",
            "I-ORGANISATION",
            "B-PERSON",
            "I-PERSON",
            "B-PERCENTAGE",
            "I-PERCENTAGE",
            "B-POSITION",
            "I-POSITION",
            "B-PRODUCT",
            "I-PRODUCT",
            "B-PROJECT",
            "I-PROJECT",
            "B-QUANTITY",
            "I-QUANTITY",
            "B-TIME",
            "I-TIME",
        ];
        SimplifiedTagsBatch = [];
        for NERTagsWord in NERTagsSequence:
            TagString = OriginalNERTagsNames[NERTagsWord]
            if TagString.startswith(('B-', 'I-')):
                SimplifiedTagsBatch.append(TagString[2:]);
        return SimplifiedTagsBatch;

    def AddClassificationFlags(self):
        for Split in self.HuggingFaceDataset.keys():
            self.HuggingFaceDataset[Split] = self.HuggingFaceDataset[Split].map(
                lambda Sample: {
                    'time_sensitive': DetermineTimeSensitiveLabel(self.PreprocessNerTags(Sample['ner_tags'])),
                    'organizational_interaction': DetermineOrganizationalInteractionLabel(self.PreprocessNerTags(Sample['ner_tags'])),
                    'personal_relevance': DeterminePersonalRelevanceLabel(self.PreprocessNerTags(Sample['ner_tags'])),
                    'commercial_intent': DetermineCommercialIntentLabel(self.PreprocessNerTags(Sample['ner_tags']))
                },
                batched=False
            );
    def GetHuggingFaceDatasets(self):
        return self.HuggingFaceDataset['train'], self.HuggingFaceDataset['validation'], self.HuggingFaceDataset['test'];

    def GetDatasets(self, ModelType='BERT', BatchSize=32):
        def Encode(Text):
            if ModelType in ['BERT', 'GPT']:
                if ModelType == 'BERT':
                    Tokenizer = self.BertTokenizer
                elif ModelType == 'GPT':
                    Tokenizer = self.GptTokenizer
                    Tokenizer.pad_token = Tokenizer.eos_token
                Encoding = Tokenizer.encode_plus(
                    Text,
                    add_special_tokens=True,
                    max_length=128,
                    return_token_type_ids=False,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='np',
                )
                return Encoding['input_ids'], Encoding.get('attention_mask', None)
            elif ModelType == 'LSTM':
                # Assuming Token2Index mapping is available for LSTM
                Tokens = Text.split()  # Basic tokenization based on whitespace
                EncodedTokens = np.array([self.Token2Index.get(token, 0) for token in Tokens], dtype=np.int32)
                return np.pad(EncodedTokens, (0, 128 - len(EncodedTokens)), mode='constant'), None  # Padding to match fixed length
        def EncodeSamples(Sample):
            Text = ' '.join(Sample['tokens'])
            input_ids, attention_mask = Encode(Text)
            Labels = [
                Sample['time_sensitive'],
                Sample['organizational_interaction'],
                Sample['personal_relevance'],
                Sample['commercial_intent']
            ]
            return (input_ids, attention_mask), np.array(Labels, dtype=np.int32)
        def GenerateDataset(DatasetSplit, include_attention_mask=True):
            Samples, Labels = [], []
            for Sample in DatasetSplit:
                (input_ids, attention_mask), _Labels = EncodeSamples(Sample)
                if include_attention_mask:
                    Samples.append((input_ids, attention_mask))
                else:
                    Samples.append(input_ids)
                Labels.append(_Labels)
            LabelsTensor = np.stack(Labels, axis=0)
            if include_attention_mask:
                InputIdsTensor = np.vstack([Sample[0] for Sample in Samples])
                AttentionMaskTensor = np.vstack([Sample[1] for Sample in Samples])
                Dataset = tf.data.Dataset.from_tensor_slices((
                    (tf.constant(InputIdsTensor, dtype=tf.int32), tf.constant(AttentionMaskTensor, dtype=tf.int32)),
                    tf.constant(LabelsTensor, dtype=tf.int32)
                ))
            else:
                InputIdsTensor = np.vstack(Samples)
                Dataset = tf.data.Dataset.from_tensor_slices((
                    tf.constant(InputIdsTensor, dtype=tf.int32),
                    tf.constant(LabelsTensor, dtype=tf.int32)
                ))
            Dataset = Dataset.batch(BatchSize)
            Dataset = Dataset.prefetch(tf.data.experimental.AUTOTUNE)
            return Dataset
        IncludeAttentionMask = True;
        if ModelType == 'LSTM':
            IncludeAttentionMask = False;
        TrainSet = GenerateDataset(self.HuggingFaceDataset['train'], include_attention_mask=IncludeAttentionMask);
        ValidSet = GenerateDataset(self.HuggingFaceDataset['validation'], include_attention_mask=IncludeAttentionMask);
        TestSet = GenerateDataset(self.HuggingFaceDataset['test'], include_attention_mask=IncludeAttentionMask);
        return TrainSet, ValidSet, TestSet;


if __name__ == "__main__":
    DataPrepare = DataPreparation()
    TrainSet, ValidSet, TestSet = DataPrepare.GetHuggingFaceDatasets();
    def SummarizeDataset(DatasetSplit, SplitName):
        TotalSequences = len(DatasetSplit);
        UniqueTokens = set(token for sample in DatasetSplit for token in sample['tokens']);
        VocabularySize = len(UniqueTokens);
        TimeSensitiveCount = sum(sample['time_sensitive'] for sample in DatasetSplit)
        OrganizationalInteractionCount = sum(sample['organizational_interaction'] for sample in DatasetSplit)
        PersonalRelevanceCount = sum(sample['personal_relevance'] for sample in DatasetSplit)
        CommercialIntentCount = sum(sample['commercial_intent'] for sample in DatasetSplit)
        print(f"--- {SplitName.upper()} SET SUMMARY ---");
        print(f"Total Sequences: {TotalSequences}");
        print(f"Vocabulary Size: {VocabularySize}");
        print("Label Counts:");
        print(f"\tTime-Sensitive: {TimeSensitiveCount}");
        print(f"\tOrganizational Interaction: {OrganizationalInteractionCount}");
        print(f"\tPersonal Relevance: {PersonalRelevanceCount}");
        print(f"\tCommercial Intent: {CommercialIntentCount}");
        print("\n");

    SummarizeDataset(TrainSet, "train");
    SummarizeDataset(ValidSet, "validation");
    SummarizeDataset(TestSet, "test");


