from preprocessor import Preprocessor
import re

class QITEmailBodyCleaner(Preprocessor):

    def __init__(self, preprocessor):

        self.preprocessor = preprocessor


    def preprocess(self):

        preprocessed_sentences = self.preprocessor.preprocess()
        return list( map(self._clean_email_body, preprocessed_sentences) )


    # Converte una stringa del tipo '=AB=CD' nel corrispettivo carattere unicode.
    # Ad esempio, la stringa '=C3=A8' diventa 'è' (il suo codice esadecimale utf-8 è 0xc3a8)
    def _convert_to_unicode(self, string):

        string = string.replace('=', '')
        string = string.lower()
        try:
            string = bytes.fromhex(string).decode('utf-8')
        except:
            string = ''

        return string


    def _clean_email_body(self, raw_email_body):

        # A volte compaiono dei caratteri '=' che precedono gli '\n'. Li rimuovo
        raw_email_body = raw_email_body.replace('=\n', '')
        raw_email_body = raw_email_body.replace('\n', ' ')
        raw_email_body = raw_email_body.replace('\t', ' ')

        # Espressione regolare per catturare stringhe del tipo =AB, =AB=CD o =AB=CD=EF
        regex_utf8_hex_values = re.compile(r'(=(\w\w)?)+')

        # Sostituisco le stringhe trovate con i corrispettivi caratteri unicode
        decoded_email_body = regex_utf8_hex_values.sub(repl=lambda match: self._convert_to_unicode( match.group() ), string=raw_email_body)

        # Elimino gli spazi in eccesso
        decoded_email_body = re.sub(pattern=r'\s(\s)+', repl=' ', string=decoded_email_body)
        decoded_email_body = decoded_email_body.strip()

        return decoded_email_body
