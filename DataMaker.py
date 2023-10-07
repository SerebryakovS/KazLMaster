

def ReadData(config):
    word_data = open('data/word/train.txt', 'r').read().replace('\n', '<eos>').split()
    words = list(set(word_data))

    word_data_size, word_vocab_size = len(word_data), len(words)
    print('data has %d words, %d unique' % (word_data_size, word_vocab_size))
    # config.word_vocab_size = word_vocab_size
    word_to_ix = { word:i for i,word in enumerate(words) }
    ix_to_word = { i:word for i,word in enumerate(words) }

    def get_word_raw_data(input_file):
      data = open(input_file, 'r').read().replace('\n', '<eos>').split()
      return [word_to_ix[w] for w in data]

    train_raw_data = get_word_raw_data('data/word/train.txt')
    valid_raw_data = get_word_raw_data('data/word/valid.txt')
    test_raw_data = get_word_raw_data('data/word/test.txt')

    return word_vocab_size, train_raw_data, valid_raw_data, test_raw_data

if __name__ == "__main__":
