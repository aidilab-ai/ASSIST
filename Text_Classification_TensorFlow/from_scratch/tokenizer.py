from keras.preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize

class IntegerTokenizer:

    def __init__(self, vocabulary):

        self.vocabulary = vocabulary
        self.unknown_token_id = max(vocabulary.values()) + 1
        self.padding_token_id = max(vocabulary.values()) + 2


    def _integer_encode(self, ticket, language):

        string_tokens = word_tokenize(ticket, language=language)
        integer_tokens = map(lambda str_tok: self.vocabulary.get(str_tok, self.unknown_token_id), string_tokens)
        return list(integer_tokens)


    def tokenize_and_pad(self, tickets, sequence_length, language='italian'):

        tokenized_tickets = list( map(lambda ticket: self._integer_encode(ticket, language=language), tickets) )
        tokenized_tickets = pad_sequences(tokenized_tickets,
                                          value=self.padding_token_id,
                                          padding='post',
                                          maxlen=sequence_length)
        return tokenized_tickets
