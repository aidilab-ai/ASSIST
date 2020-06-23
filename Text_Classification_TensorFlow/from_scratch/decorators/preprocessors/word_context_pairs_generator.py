from preprocessor import Preprocessor

class WordContextPairsGenerator(Preprocessor):

    def __init__(self, preprocessor, window_length):

        self.preprocessor = preprocessor
        self.window_length = window_length


    def preprocess(self):

        preprocessed_sentences = self.preprocessor.preprocess()

        pairs = []
        for sentence in preprocessed_sentences:

            num_words = len(sentence)
            for i in range(self.window_length, num_words - self.window_length):

                preceding_words = sentence[i - self.window_length : i]
                following_words = sentence[i + 1 : i + 1 + self.window_length]
                pairs += [(sentence[i], p) for p in preceding_words] + [(sentence[i], f) for f in following_words]

        return pairs
