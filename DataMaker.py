import os
import re
import time
from ProcessingTools.kaznlp.lid.lidnb import LidNB
from ProcessingTools.kaznlp.tokenization.tokrex import TokenizeRex

################################################################################################
def GetDataSetsDefault(DefaultDataPath="./data/defaultData/word"):
    word_data = open(f"{DefaultDataPath}/train.txt", 'r').read().replace('\n', '<eos>').split()
    words = list(set(word_data))
    word_data_size, word_vocab_size = len(word_data), len(words)
    print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
    word_to_ix = { word:i for i,word in enumerate(words) }
    ix_to_word = { i:word for i,word in enumerate(words) }
    def get_word_raw_data(input_file):
        data = open(input_file, 'r').read().replace('\n', '<eos>').split()
        return [word_to_ix[w] for w in data]
    train_raw_data = get_word_raw_data(f"{DefaultDataPath}/train.txt")
    valid_raw_data = get_word_raw_data(f"{DefaultDataPath}/valid.txt")
    test_raw_data = get_word_raw_data(f"{DefaultDataPath}/test.txt")
    return word_vocab_size, {"Train":train_raw_data, "Valid": valid_raw_data, "Test": test_raw_data}
################################################################################################

def PrepareRawDataFiles(SetsPath, OutputFilepath):
    # parallel and async execution !!!
    KazakhAlphabet = u"аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцшъыіьэАӘБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФХҺЦШЪЫІЬЭ";
    TokRex = TokenizeRex();
    BasePath = os.getcwd(); TrainSets = os.listdir(SetsPath);
    LanguageDetector = LidNB(word_mdl=None, char_mdl=os.path.join('ProcessingTools','kaznlp', 'lid', 'char.mdl'));
    for SetFolder in TrainSets:
        os.chdir(SetsPath+SetFolder);
        for SingleTextFilename in os.listdir():
            try:
                with open(SingleTextFilename,"r") as OpenedFilePtr:
                    FileContent = OpenedFilePtr.readlines();
                    FileContent = re.sub('\s+',' ', "".join(FileContent)).lower();  # remove all whitespace characters
                    FileContent = re.sub(r'\b\w\b', '', FileContent);               # remove single character words
                    NewFileContent = str(); PrevWord = "";
                    for Idx, Word in enumerate(TokRex.tokenize(FileContent)[0]):
                        WordLanguage = LanguageDetector.predict(Word);
                        if WordLanguage == "other":
                            Word = "<pun>";
                        elif WordLanguage != "kazakh":
                            Word = "<unk>";
                        if WordLanguage != "kazakh" and PrevWord == Word:
                            continue;
                        NewFileContent += f"{Word} "
                        PrevWord = Word;
                    NewFileContent = "\n".join(list(set(NewFileContent.replace("<pun>","\n").split("\n"))));
                    # remove empty lines if exist also !!!!
                    with open(OutputFilepath, 'a') as OutputFile:
                        OutputFile.write(NewFileContent);
            except Exception as Ex:
                print(f"[ERR]: Exception:{Ex}");
        os.chdir(BasePath);

def PrepareVocabulary(TrainDataFilename):
    RawTrainData = [];
    with open(TrainDataFilename, 'r') as TrainDataFile:
        for SingleLine in TrainDataFile:
            RawTrainData += SingleLine.replace('\n', '<eos>').split();

    # RawTrainData = open(TrainDataFilename, 'r').read().replace('\n', '<eos>').split();


    UniqueWords = list(set(RawTrainData));
    TotalWordsCount, VocabularySize = len(RawTrainData), len(UniqueWords);
    print('[INFO]: Data has %d words, %d unique' % (TotalWordsCount, VocabularySize));
    word_to_ix = { Word:Idx for Idx,Word in enumerate(UniqueWords) };
    def GetReadySet(InputFilename):
        RawData = open(InputFilename, 'r').read().replace('\n', ' <eos> ').split();
        MarkedData = list();
        for Word in RawData:
            try:
                MarkedData.append(word_to_ix[Word]);
            except KeyError:
                print(f"[WARN][{InputFilename}]: Unknown word found: {Word}");
                MarkedData.append(word_to_ix["<unk>"]);
        return MarkedData;
    return VocabularySize, GetReadySet;








