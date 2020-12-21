import TextPreprocessing
import embeddings
import clustering
import spacy
import pandas as pd
from deepsegment import DeepSegment
segmenter = DeepSegment('en')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

def split_into_sentences(utterance):
    """
    Splits a string or SpaCy doc into sentences.
    
    (You must include 'setencizer' in the SpaCy NLP pipeline 
    so that you can split the sentences out.)
    
    :utterance: string of text or SpaCy doc
    :return: list of strings or SpaCy spans of sentences.
    """
    if isinstance(utterance, str):
        if len(utterance) < 1:
            return ['']
        elif len(utterance) > 200:
            sents = segmenter.segment_long(utterance,n_window=10)
        else:
            sents = segmenter.segment(utterance)
        return sents
    else:
        return [sentence for sentence in utterance.sents]

def get_sentence_count(utterance):
    """
    Gets the number of sentences in a given utterance.

    :utterance: string of text or SpaCy doc
    :return: and integer >= 0 indicating the number of sentences.
    """
    return len(split_into_sentences(utterance))
    
def get_word_count(utterance):
    """
    This function counts the number of word tokens in an utterance.

    :utterance: string of text or SpaCy doc 
    :return: integer >= 0 for the total number of tokens
    """
    if isinstance(utterance, str):
        return sum(1 for word in utterance.split())
    else: 
        return sum(1 for word in utterance.text.split())

def get_word_density(utterance):
    """Calculates the average length of the words in the utterance.

    :utterance: SpaCy doc
    :return: integer >= 0 for the average word length.
    """
    total_characters = 0
    for token in utterance:
        if not token.is_punct and token.is_alpha:
            total_characters += len(token.text)
    return total_characters / get_word_count(utterance)

def get_punctuation_count(utterance):
    """
    Returns the number of punctuations in the input utterance.

    :utterance: SpaCy doc
    :return: integer >= 0 for the number of punctuations.
    """
    return sum([1 for token in utterance if token.is_punct])

def get_uppercase_count(utterance):
    """
    Gets the number of uppercase letters in an input SpaCy doc.

    :utterance: SpaCy doc
    :return: integer >= 0 for the number of uppercase letters.
    """
    return sum(1 for c in utterance.text if c.isupper())

def get_lowercase_count(utterance):
    """
    Gets the number of lowercase letters in an input SpaCy doc.

    :utterance: SpaCy doc
    :return: integer >= 0 for the number of lowercase letters.
    """
    return sum(1 for c in utterance.text if c.islower())

def get_case_ratio(utterance):
    """Gets the ratio of uppercase letters.

    :utterance: SpaCy doc
    :return: float >= 0 for the ratio of uppercase letters to lowercase letters.
    """

    uppercase = get_uppercase_count(utterance)
    lowercase = get_lowercase_count(utterance)
    return uppercase / (uppercase + lowercase)

def get_sentiment(utterance):
    """
    Returns the sentiment polarity between -1 and 1 for how
    negative or positive the utterance is. 

    This function uses NLTK's VADER lexicon approach for getting the 
    sentiment score.

    :utterance: SpaCy doc
    :return: -1 >= float <= 1 for utterance sentiment.
    """
    return sid.polarity_scores(utterance.text)['compound']

class text_EDA():

    def __init__(self, utterances, pipes = ['entity_ruler', 'sentencizer']) -> None:
            self.data = pd.DataFrame(utterances, columns=['Raw Utterances'])
            self.nlp_utterances = None

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
        if self.nlp_utterances == None:
            self.nlp_utterances = list(self.nlp.pipe(self.data['Raw Utterances']))
        self.data['Sentence Counts'] = list(map(get_sentence_count, self.nlp_utterances))
        self.data['Word Counts'] = list(map(get_word_count, self.nlp_utterances))
        self.data['Word Densities'] = list(map(get_word_density, self.nlp_utterances))
        self.data['Punctuation Counts'] = list(map(get_punctuation_count, self.nlp_utterances))
        self.data['Case Ratios'] = list(map(get_case_ratio, self.nlp_utterances))