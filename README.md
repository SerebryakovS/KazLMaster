## Kazakh language NLP processing tools/models collection

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



Notes:
1. Common practices for setting the number of epochs include monitoring the model's performance on a validation dataset and stopping training early (early stopping) if the performance no longer improves. It helps prevent overfitting and ensures that the model generalizes well to unseen data.

2. The whole program operation could be pipelined.

3. During the computing of probabilities of the word sequences it’s important to define the boundaries (punctuation marks such as period, comma, column or starting of the new sentence from the new line) in order to prevent the search from being computationally unmanageable.

### Literature:

    1. learning rate: https://wiki.loginom.ru/articles/learning-rate.html
