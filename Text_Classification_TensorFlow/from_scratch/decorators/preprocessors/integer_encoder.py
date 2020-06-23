from preprocessor import Preprocessor

class IntegerEncoder(Preprocessor):

    def __init__(self, preprocessor, vocabulary, unknown_token_id):

        self.preprocessor = preprocessor
        self.vocabulary = vocabulary
        self.unknown_token_id = unknown_token_id


    def _integer_encode(self, tokens):

        integer_tokens = map(lambda tok: self.vocabulary.get(tok, self.unknown_token_id), tokens)
        return list(integer_tokens)


    def preprocess(self):

        preprocessed_sentences = self.preprocessor.preprocess()
        return list( map(self._integer_encode, preprocessed_sentences) )
