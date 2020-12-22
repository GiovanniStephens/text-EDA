import embeddings
import clustering
import spacy
import pytextrank
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def split_into_sentences(utterance):
    """
    Splits a SpaCy doc into sentences.
    
    (You must include 'setencizer' in the SpaCy NLP pipeline 
    so that you can split the sentences out.)
    
    :utterance: SpaCy doc
    :return: list of SpaCy spans of sentences.
    """
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

def get_cluster_labels(utterances):
    """
    Calculates cluster labels for all the utterances. 
    If there are few documents, I use PCA for dimension reduction and k-means
    for clustering. If there are more than 100, I use UMAP and HDBSCAN together.

    :utterances: list of strings
    :return: list of integers indicating what cluster they belong to.
    """
    n_docs = len(utterances)
    if n_docs <= 100:
        document_embeddings = embeddings.pretrained_transformer_embeddings(utterances)
        dims = min(clustering.get_optimal_n_components(document_embeddings), 10)
        reduced = clustering.reduce_dimensions_pca(document_embeddings,\
            dimensions=dims)
        return clustering.kmeans_clustering(reduced,\
            max_num_clusters=min(n_docs,80))
    else:
        document_embeddings = embeddings.word2vec_sif_embeddings(utterances,model_name=None)
        reduced = clustering.reduce_dimensions_umap(document_embeddings)
        return clustering.hdbscan_clustering(reduced)

def get_top_words(utterances, n = 20):
    """
    This function returns the top n words in a list SpaCy docs.

    :utterances: list of SpaCy docs.
    :return: 
    """
    words = []
    for utterance in utterances:
        for word in utterance:
            if word.is_alpha:
                words.append(word.text)
    counter = Counter(words)
    top_words = counter.most_common(n)  
    return top_words

def get_top_n_ngrams(corpus, N = 1, n=None):
    """
    This function gives the n top N-grams along with their counts in desc order. 
    """

    vec = CountVectorizer(ngram_range=(N, N)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_top_pos(utterances, pos = 'VERB', n = 20):
    """
    This function takes a list of lists of word tokens, the type of phrase position
    (i.e. verb, noun, etc.) and the n number of top words of the pos type. It goes through the
    lists of lists, joins the tokens, and works out which words are of that pos. It will then 
    count the number of occurances in that word. 
    Returns lemma of top n words that are of type pos.
    """

    words = []
    for utterance in utterances:
        for word in utterance:
            if word.pos_ == pos:
                words.append(word.lemma_)
    counter = Counter(words)
    top_words = counter.most_common(n)  
    return top_words

def get_top_entities(utterances, entity_types, n = 20):
    entities = []
    for utterance in utterances:
        for word in utterance:
            if word.ent_type_ in entity_types:
                entities.append(word.text)
    counter = Counter(entities)
    top_entities = counter.most_common(n)  
    return top_entities

def get_top_keywords(keywords, n = 20):
    words = []
    for doc_keywords in keywords:
        for keyword in doc_keywords:
            words.append(keyword.lemma_)
    counter = Counter(words)
    top_words = counter.most_common(n)  
    return top_words


class text_EDA():

    def __init__(self, utterances, pipes = ['entity_ruler', 'sentencizer', 'textrank']) -> None:
            self.data = pd.DataFrame(utterances, columns=['Raw Utterances'])
            self.top_features = None
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
            if pipe == 'sentencizer': #needs to go before the parser. 
                nlp_pipe = self.nlp.create_pipe(pipe)
                self.nlp.add_pipe(nlp_pipe, before='parser')
            elif pipe == 'textrank':
                tr = pytextrank.TextRank()
                self.nlp.add_pipe(tr.PipelineComponent, name=pipe, last=True)
            else:
                nlp_pipe = self.nlp.create_pipe(pipe)
                self.nlp.add_pipe(nlp_pipe)

    def explore(self):
        if self.nlp_utterances == None:
            self.nlp_utterances = list(self.nlp.pipe(self.data['Raw Utterances']))
        self.data['Sentence Counts'] = list(map(get_sentence_count, self.nlp_utterances))
        self.data['Word Counts'] = list(map(get_word_count, self.nlp_utterances))
        self.data['Word Densities'] = list(map(get_word_density, self.nlp_utterances))
        self.data['Punctuation Counts'] = list(map(get_punctuation_count, self.nlp_utterances))
        self.data['Case Ratios'] = list(map(get_case_ratio, self.nlp_utterances))
        self.data['Sentiments'] = list(map(get_sentiment, self.nlp_utterances))
        self.data['Categories'] = get_cluster_labels(self.data['Raw Utterances'])

        top_words = pd.DataFrame(get_top_words(self.nlp_utterances), \
            columns=['Top Words', 'Top Words Counts'])
        
        n_grams = []
        corpus = [utterance.text for utterance in self.nlp_utterances]
        for n in range(1,3):
            n_grams.append(pd.DataFrame(get_top_n_ngrams(corpus,N=n,n=20), columns=[f'Top n-grams ({n})', 'Top n-gram Counts']))

        top_nouns = pd.DataFrame(get_top_pos(self.nlp_utterances, pos='NOUN', n=20),\
            columns=['Top Nouns', 'Top Nouns Counts'])
        top_verbs = pd.DataFrame(get_top_pos(self.nlp_utterances, pos='VERB', n=20),\
            columns=['Top Verbs', 'Top Verbs Counts'])
        top_people = pd.DataFrame(get_top_entities(self.nlp_utterances, ['PERSON'], 20),\
            columns=['Top People', 'Top People Counts'])
        top_organisations = pd.DataFrame(get_top_entities(self.nlp_utterances, ['ORG', 'NORP'], 20),\
            columns=['Top Oraganisations', 'Top Oraganisations Counts'])
        top_locations = pd.DataFrame(get_top_entities(self.nlp_utterances, ['FAC', 'GPE', 'LOC'], 20),\
            columns=['Top Locations', 'Top Locations Counts'])

        keyphrases_corpus = [[keyphrase.chunks[0] for keyphrase in utterance._.phrases] for utterance in self.nlp_utterances]
        top_keyphrases = pd.DataFrame(get_top_keywords(keyphrases_corpus,n=20), columns=['Top Keyphrases', 'Top Keyphrases Counts'])
        self.top_features = top_words
        for n_gram in n_grams:
            self.top_features = pd.concat([self.top_features, n_gram], axis=1)
        self.top_features = pd.concat([self.top_features, top_nouns], axis=1)
        self.top_features = pd.concat([self.top_features, top_verbs], axis=1)
        self.top_features = pd.concat([self.top_features, top_people], axis=1)
        self.top_features = pd.concat([self.top_features, top_organisations], axis=1)
        self.top_features = pd.concat([self.top_features, top_locations], axis=1)
        self.top_features = pd.concat([self.top_features, top_keyphrases], axis=1)