import TextPreprocessing
import embeddings
import clustering
import spacy
import pandas as pd

def get_word_count(utterance):
    """
    This function counts the number of word tokens in an utterance.

    :utterance: string of text
    :return: integer >= 0 for the total number of tokens
    """
    if isinstance(utterance, str):
        return sum(1 for word in utterance.split())
    else: 
        return sum(1 for word in utterance.text.split())

class text_EDA():

    def __init__(self, utterances, pipes = ['entity_ruler', 'sentencizer']) -> None:
            self.data = pd.DataFrame(utterances, columns=['Raw Utterances'])

            # Load SpaCy model
            self.nlp = spacy.load('en_core_web_sm')
            # Load SpaCy pipeline 
            if pipes != None:
                self.load_nlp_pipe(pipes)
        
    # Load NLP pipeline
    def load_nlp_pipe(self, pipes):
        """
        This function creates and loads all the pipes into the nlp-er

        :pipes: list of pipe names as strings
        """

        for pipe in pipes:
            nlp_pipe = self.nlp.create_pipe(pipe)
            if pipe == 'sentencizer': #needs to go before the parser. 
                self.nlp.add_pipe(nlp_pipe, before='parser')
            else:
                self.nlp.add_pipe(nlp_pipe)

    def explore(self):
        self.data['Word Counts'] = map(get_word_count, self.data['Raw Utterances'])
