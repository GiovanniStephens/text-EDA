import TextPreprocessing
import embeddings
import clustering

def get_word_count(utterance):
    """
    This function counts the number of word tokens in an utterance.

    :utterance: string of text
    :return: integer >= 0 for the total number of tokens
    """
    
    return sum(1 for word in utterance.split())