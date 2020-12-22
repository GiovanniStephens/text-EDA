import unittest

TEST_UTTERANCES = (
    'This is a test sentence.'
    , 'This is a similar test phrase.'
    , 'Nothing to do with the others.'
    , 'This is a multi-sentence phrase. I want to know if I can split them.'
    , 'This is another test sentence for me I want to test whether I can split them when there is no punctuation'
    , 'My dr. is testing me.'
    , 'He owes me $200 bucks!'
    , 'That conversation did not go well.'
    , 'I am pretty happy with that result.'
    , 'Is this testing phrase going to put with the others?'
    , 'This is another testing phrase to be put together with the others.'
    , 'This testing phrase is similar.'
    , 'John is my friend.'
    , 'John works at Google with Mandy in Florida.'
)

class test_embeddings(unittest.TestCase):

    def _get_module(self, module_name):
        import importlib
        return importlib.import_module(module_name)

    def test_load_nlp_pipe(self):
        """Tests that the nlp pipe has been set up correctly."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        self.assertEqual(explorer.nlp.pipe_names, \
             ['tagger', 'sentencizer', 'parser', 'ner', 'entity_ruler', 'textrank']) 

    def test_get_word_count_str(self):
        """Tests that we're getting the right word count from a string."""
        text_EDA = self._get_module('text_EDA')
        self.assertEqual(text_EDA.get_word_count(TEST_UTTERANCES[0]), 5)

    def test_get_word_count(self):
        """Tests running the data through SpaCy's NLP pipeline
        and then getting the word count."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        self.assertEqual(text_EDA.get_word_count(\
            explorer.data['Raw Utterances'].iloc[0]), 5)

    def test_split_into_sentences(self):
        """Test SpaCy's ability to split a string into parts."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        sents = text_EDA.split_into_sentences(explorer.nlp_utterances[3])
        self.assertEqual(len(sents), 2)

    def test_get_sentence_count(self):
        """Test getting a sentence count from a SpaCy doc object."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        cnt = text_EDA.get_sentence_count(explorer.nlp_utterances[3])
        self.assertEqual(cnt, 2)

    def test_explore_sentence_count(self):
        """Tests that the explore function is getting the sentence counts."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.data['Sentence Counts'].iloc[3],2)

    def test_get_word_density(self):
        """Tests getting the average word length for an utterance."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_word_density(\
            explorer.nlp_utterances[0]), 3.8)

    def test_get_word_density_2(self):
        """Tests getting the average word length for an utterance.
        It should not include punctuation in that count."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_word_density(\
            explorer.nlp_utterances[5]), 3)

    def test_explore_get_word_density(self):
        """Tests that the explore function is getting the word densities"""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.data['Word Densities'].iloc[5],3)

    def test_get_punctuation_count(self):
        """Tests that it is capturing the period at the end of the sentence."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_punctuation_count(\
            explorer.nlp_utterances[0]), 1)

    def test_get_punctuation_count_2(self):
        """Tests that the punctuation after the dr is also counted."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_punctuation_count(\
            explorer.nlp_utterances[5]), 2)

    def test_get_punctuation_count_3(self):
        """Tests that $ signs are not counted as punctuation."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_punctuation_count(\
            explorer.nlp_utterances[6]), 1)

    def test_explore_punctuation_count(self):
        """Tests that the explore function is getting the punctuation counts"""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.data['Punctuation Counts'].iloc[5], 2)

    def test_get_uppercase_count(self):
        """Tests that I get the right number of uppercase letters."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_uppercase_count(\
            explorer.nlp_utterances[3]), 3)

    def test_get_lowercase_count(self):
        """Tests that I get the right number of lowercase letters."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_lowercase_count(\
            explorer.nlp_utterances[0]), 18)

    def test_get_case_ratio(self):
        """Tests that I am getting the right uppercase to lowercase
        ratio."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_case_ratio(\
            explorer.nlp_utterances[0]), 1/19)

    def test_explore_get_case_ratio(self):
        """Tests that the explore function is getting the case ratios correctly"""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.data['Case Ratios'].iloc[0], 1/19)

    def test_get_sentiment_neg(self):
        """Tests the VADER sentiment analysis function on a negative phrase."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertLess(text_EDA.get_sentiment(\
            explorer.nlp_utterances[7]), 0)

    def test_get_sentiment_pos(self):
        """Tests the VADER sentiment analysis function on a positive phrase."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertGreater(text_EDA.get_sentiment(\
            explorer.nlp_utterances[8]), 0)

    def test_explore_get_sentiment_neg(self):
        """Tests that the explore function is getting the case ratios correctly"""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertLess(explorer.data['Sentiments'].iloc[7], 0)

    def test_labelling_documents(self):
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES[0:3])
        explorer.explore()
        self.assertNotEqual(explorer.data['Categories'].iloc[0], \
            explorer.data['Categories'].iloc[2])

    def test_get_top_words(self):
        """Tests that I am correctly counting the top word. 
        Is comes up 8 times."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_top_words(\
            explorer.nlp_utterances)[0][1], 9)

    def test_get_top_n_ngrams(self):
        """Tests that I am correctly counting the top bi-grams. 
        Is comes up 5 times."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        corpus = [utterance.text for utterance in explorer.nlp_utterances]
        self.assertEqual(text_EDA.get_top_n_ngrams(corpus,N=2,n=20)[0][1],\
            5)

    def test_get_top_nouns(self):
        """Tests that I am correctly counting the top nouns."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        corpus = [utterance.text for utterance in explorer.nlp_utterances]
        self.assertEqual(text_EDA.get_top_pos(\
            explorer.nlp_utterances, pos='NOUN', n=20)[0][1], 5)

    def test_get_top_entities_person(self):
        """Tests that I am correctly counting the top entities."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(text_EDA.get_top_entities(\
            explorer.nlp_utterances, ['PERSON'], 1)[0][0], 'John')

    def test_explore_get_top_nouns(self):
        """Tests that I am correctly counting the top nouns in the explore function."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.top_features['Top Nouns'].iloc[0], 'phrase')

    def test_explore_get_top_words(self):
        """Tests that I am correctly counting the top words in the explore function."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.top_features['Top Words'].iloc[0], 'is')

    def test_explore_get_top_people(self):
        """Tests that I am correctly counting the top people in the explore function."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.top_features['Top People'].iloc[0], 'John')

    def test_explore_get_top_n_grams(self):
        """Tests that I am correctly counting the top n-grams in the explore function."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.top_features['Top n-grams (2)'].iloc[0], 'this is')

    def test_explore_get_top_locations(self):
        """Tests that I am correctly counting the top locations in the explore function."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        explorer.explore()
        self.assertEqual(explorer.top_features['Top Locations'].iloc[0], 'Florida')

if __name__ == '__main__':
    unittest.main()