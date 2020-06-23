from keras.preprocessing.sequence import pad_sequences
from preprocessor import Preprocessor

class Padder(Preprocessor):

    def __init__(self, preprocessor, padding_token_id, max_length):

        self.preprocessor = preprocessor
        self.padding_token_id = padding_token_id
        self.max_length = max_length


    def preprocess(self):

        preprocessed_sentences = self.preprocessor.preprocess()
        return pad_sequences(preprocessed_sentences, value=self.padding_token_id, padding='post', maxlen=self.max_length)
