from nltk.tokenize import word_tokenize
from preprocessor import Preprocessor

class Tokenizer(Preprocessor):

    def __init__(self, preprocessor, language='italian'):

        self.preprocessor = preprocessor
        self.language = language


    def preprocess(self):

        preprocessed_sentences = self.preprocessor.preprocess()
        return list( map(lambda sentence: word_tokenize(sentence, language=self.language), preprocessed_sentences) )
