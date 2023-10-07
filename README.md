# LSTM-LM

Neural LSTM Language Modeling experiments for Kazakh

This is the fork from the original LSTM Kazakh language model presented by: https://github.com/Baghdat/LSTM-LM.git

Key difference is migration to TensorFlow V2 using V1's still supported API.

The initial step in this process is as follows:
```
tf_upgrade_v2 --infile RNN_LSTM.py --outfile RNN_LSTM_v2.py
```
This step converts a significant portion of the V1 code into a compatible V2 format.
After that some tiny changes in code were made. For example:
```
###########################################################################
# def model_size():
#   '''finds the total number of trainable variables a.k.a. model size'''
#   params = tf.compat.v1.trainable_variables()
#   size = 0
#   for x in params:
#     sz = 1
#     for dim in x.get_shape():
#       sz *= dim.value
#     size += sz
#   return size
def model_size():
    '''finds the total number of trainable variables a.k.a. model size'''
    params = tf.compat.v1.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.shape:
            sz *= dim
        size += sz
    return size
###########################################################################
```
Follow this installation guide to make it work over GPU under miniconda3 package manager:
```
https://www.tensorflow.org/install/pip
```


Notes:
1. Common practices for setting the number of epochs include monitoring the model's performance on a validation dataset and stopping training early (early stopping) if the performance no longer improves. It helps prevent overfitting and ensures that the model generalizes well to unseen data.

