## Kazakh language NLP processing tools/models collection


### INSTALLATION PROCESS
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
git clone --recursive git@github.com:SerebryakovS/KazLMaster.git && cd KazLMaster
source RunConda.sh # or: ~/miniconda3/bin/conda init bash
conda create -n nlpenv tensorflow-gpu 'transformers[tf-gpu]'
conda activate nlpenv
conda install numpy=1.19.5
conda install keras
```

### Neural LSTM Language Modeling experiments for Kazakh (fork)

This project is the fork from the original LSTM Kazakh language model presented by: https://github.com/Baghdat/LSTM-LM.git
Results were published in: https://ceur-ws.org/Vol-2303/short2.pdf
Key difference is migration to TensorFlow V2 using V1's still supported API.
The initial step in this process is as follows:
```
tf_upgrade_v2 --infile RNN_LSTM.py --outfile RNN_LSTM_v2.py
```
This step converts a significant portion of the V1 code into a compatible V2 format.
After that some tiny changes in code were made to make it work in default way.

### KazNLP: NLP tools for Kazakh language (submodule)

The default processing tools are set to the following project in use: https://github.com/makazhan/kaznlp.git
Due to the need to train large amounts of information in the Kazakh language, a ready-made tool was used, including models and useful preprocessing functions out of the box

### Configuration feature

This project highly extends functionality of the original one. Key feature is the configuration flexibility. In the included OperationParameters.json file user can set desired operation mode and stages he interested in. For example, if u've already trained the model before, just set TrainAndValidStage's StageEnabled parameter to False. This action will omit training stage.


### Training flow features

Here used a smarter approach of using strategies that adapt the learning rate based on the training process, such as:


    Early Stopping: Monitor the validation performance and stop training when the validation metric stops improving.
    
    Reduce on Plateau: Lower the learning rate by a certain factor when the validation metric has stopped improving.



### BERT

BERT is based on the Transformer model architecture instead of LSTMs.
A transformer model is a neural network that learns context and thus meaning by tracking relationships in sequential data like the words in this sentence. 
One key advantage of Transformers over LSTMs is their more effective handling of long-range dependencies, due to the self-attention mechanism. This allows them to weigh the importance of various positions in the input sequence, whereas LSTMs might struggle with retaining information from distant positions in longer sequences.
Both Transformers and LSTMs have shown excellent performance in tasks like machine translation, speech recognition, text classification, and more.

Text classification in NLP involves categorizing and assigning predefined labels or categories to text documents, sentences, or phrases based on their content. Text classification aims to automatically determine the class or category to which a piece of text belongs.




Notes:
1. Common practices for setting the number of epochs include monitoring the model's performance on a validation dataset and stopping training early (early stopping) if the performance no longer improves. It helps prevent overfitting and ensures that the model generalizes well to unseen data.

2. The whole program operation could be pipelined.

3. During the computing of probabilities of the word sequences it’s important to define the boundaries (punctuation marks such as period, comma, column or starting of the new sentence from the new line) in order to prevent the search from being computationally unmanageable.

A low perplexity indicates the probability distribution is good at predicting the sample.

### Literature:

    1. learning rate: https://wiki.loginom.ru/articles/learning-rate.html
    2. https://medium.com/data-science-365/all-you-need-to-know-about-batch-size-epochs-and-training-steps-in-a-neural-network-f592e12cdb0a
    3. https://www.analyticsvidhya.com/blog/2023/06/step-by-step-bert-implementation-guide/
    4. https://medium.com/@MilkKnight/build-your-lstm-language-model-with-tensorflow-3416142c9919
    5. https://medium.com/@manindersingh120996/accelerate-your-text-data-analysis-with-custom-bert-word-embeddings-and-tensorflow-45590cf9c54
    6. https://medium.com/unpackai/perplexity-and-accuracy-in-classification-114b57bd820d

### Articles:

    1. https://www.mdpi.com/2504-2289/6/4/123
    2. https://dergipark.org.tr/en/download/article-file/2490930



So numerical representation of words in order to train ML models is termed Word embedding.


### Claffification Project:

conda update numpy

We'll focus on Masked Language Modeling (MLM) as it's a common and widely applicable task for understanding and leveraging the capabilities of language models.
In MLM, the model is trained to predict the identity of masked tokens in a sequence. This task is essential for models like BERT, which learn to understand the context of a word in a sentence in a bidirectional manner.
Out dataset, kz-transformers/multidomain-kazakh-dataset, is already suitable for MLM as it's text-based. The preparation involves tokenizing the text data and creating masked tokens for the model to predict.

Creating standardized classes for both LSTM and BERT models to perform a specific NLP task, such as masked language modeling (MLM), provides a clean and efficient way to compare their performance. Given the complexity of the MLM task and the fact that LSTM models don't natively support MLM in the same way BERT and other transformer models do, we'll adapt our approach to fit a more generalized comparison framework.


