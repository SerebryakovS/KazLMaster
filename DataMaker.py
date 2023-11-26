import os
import re
import time
import asyncio
from multiprocessing import Queue, Manager
from asyncio import Semaphore
from concurrent.futures import ProcessPoolExecutor
from ProcessingTools.kaznlp.lid.lidnb import LidNB
from ProcessingTools.kaznlp.tokenization.tokrex import TokenizeRex

################################################################################################

def GetDataSetsDefault(DefaultDataPath="./data/defaultData/word"):
    word_data = open(f"{DefaultDataPath}/train.txt", 'r').read().replace('\n', '<eos>').split()
    words = list(set(word_data))
    word_data_size, word_vocab_size = len(word_data), len(words)
    print('[INFO][DEF]: Data has %d words, %d unique' % (word_data_size, word_vocab_size))
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

def MergeDataWithDefaultSet(DefaultDataPath, OutputFilepath):
    with open(OutputFilepath, 'a') as OutputFile:
        with open(f"{DefaultDataPath}/train.txt", 'r') as DefTrainDataFile:
            OutputFile.write(DefTrainDataFile.read());

def ProcessFileContents(FileContents):
    # Process the file contents (this is a CPU-bound operation)
    TokRex = TokenizeRex()
    LanguageDetector = LidNB(word_mdl=None, char_mdl=os.path.join('ProcessingTools', 'kaznlp', 'lid', 'char.mdl'))
    FileContent = re.sub('\s+', ' ', "".join(FileContents)).lower()  # remove all whitespace characters
    FileContent = re.sub(r'\b\w\b', '', FileContent)  # remove single character words
    NewFileContent = str()
    PrevWord = ""
    for Word in TokRex.tokenize(FileContent)[0]:
        WordLanguage = LanguageDetector.predict(Word)
        if WordLanguage == "other":
            Word = "<pun>"
        elif WordLanguage != "kazakh":
            Word = "<unk>"
        if WordLanguage != "kazakh" and PrevWord == Word:
            continue
        NewFileContent += f"{Word} "
        PrevWord = Word
    return "\n".join(list(set(NewFileContent.replace("<pun>", "\n").split("\n"))))

async def ProcessFile(executor, filename, SetsPath, SetFolder, queue, semaphore):
    async with semaphore:  # limit the number of concurrent tasks
        try:
            Filename = os.path.join(SetsPath, SetFolder, filename)
            print(f"[INFO]: Preparing: {Filename}")
            with open(Filename, "r") as OpenedFilePtr:
                FileContents = OpenedFilePtr.readlines()

            loop = asyncio.get_event_loop()
            NewFileContent = await loop.run_in_executor(executor, ProcessFileContents, FileContents)
            queue.put_nowait((Filename,NewFileContent))
        except Exception as Ex:
            print(f"[ERR]: Exception: {Ex}")

async def WriteToFile(queue, OutputFilepath):
    while True:
        item = await queue.get()
        if item is None:  # Check for the termination signal
            break
        Filename, NewFileContent = item
        with open(OutputFilepath, 'a') as OutputFile:
            OutputFile.write(NewFileContent)
        print(f"[INFO]: Done: {Filename}")

async def ProcessFolder(executor, SetsPath, SetFolder, OutputFilepath, semaphore, queue):
    folder_path = os.path.join(SetsPath, SetFolder)
    filenames = os.listdir(folder_path)
    tasks = [asyncio.create_task(ProcessFile(executor, filename, SetsPath, SetFolder, queue, semaphore)) for filename in filenames]
    await asyncio.gather(*tasks)

async def _PrepareRawDataFiles(SetsPath, OutputFilepath, ProcessFileSemaCount, ProcessFileContentsThreadsCount):
    Executor = ProcessPoolExecutor(max_workers=ProcessFileContentsThreadsCount)
    semaphore = Semaphore(ProcessFileSemaCount)
    queue = asyncio.Queue()

    TrainSets = os.listdir(SetsPath)
    folder_tasks = [asyncio.create_task(ProcessFolder(Executor, SetsPath, SetFolder, OutputFilepath, semaphore, queue)) for SetFolder in TrainSets]
    writer_task = asyncio.create_task(WriteToFile(queue, OutputFilepath))

    await asyncio.gather(*folder_tasks)
    await queue.put(None)  # Signal the writer task to finish
    await writer_task

    Executor.shutdown()

def PrepareRawDataFiles(SetsPath, OutputFilepath, ProcessFileSemaCount, ProcessFileContentsThreadsCount):
    try:
        os.remove(OutputFilepath);
    except OSError:
        pass
    asyncio.run(_PrepareRawDataFiles(SetsPath, OutputFilepath, ProcessFileSemaCount, ProcessFileContentsThreadsCount))
    
################################################################################################
        
def PrepareVocabulary(TrainDataFilename, Percentage=1.0):
    RawTrainData = [];
    with open(TrainDataFilename, 'r') as TrainDataFile:
        TotalLinesInFile = sum(1 for _ in TrainDataFile);
    DesiredLinesCount = int(TotalLinesInFile * Percentage);
    with open(TrainDataFilename, 'r') as TrainDataFile:
        for Idx, SingleLine in enumerate(TrainDataFile):
            if Idx >= DesiredLinesCount:
                break
            RawTrainData += SingleLine.replace('\n', '<eos>').split();
    UniqueWords = list(set(RawTrainData));
    TotalWordsCount, VocabularySize = len(RawTrainData), len(UniqueWords);
    print('[INFO][NEW]: Data has %d words, %d unique' % (TotalWordsCount, VocabularySize));
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
    return VocabularySize, [word_to_ix[word] for word in RawTrainData], GetReadySet;

##### helpers
def GetListPart(InputList, Percentage):
    if Percentage < 0 or Percentage > 1:
        raise ValueError("Percentage should be between 0 and 1");
    CountElements = len(InputList);
    NumberToTake = int(CountElements * Percentage);
    return InputList[:NumberToTake];





