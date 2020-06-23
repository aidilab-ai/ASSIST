from preprocessor import Preprocessor

class TextPreprocessor(Preprocessor):

    def __init__(self, sentences):
        self.sentences = sentences

    def preprocess(self):
        return self.sentences
