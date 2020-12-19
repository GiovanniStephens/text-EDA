import unittest

TEST_UTTERANCES = (
    'This is a test sentence.'
    , 'This is a similar test phrase.'
    , 'Nothing to do with the others.'
    , 'This is a multi-sentence phrase. I want to know if I can split them.'
    , 'This is another test sentence for me I want to test whether I can split them when there is no punctuation'
    , 'My dr. is testing me.'
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
             ['tagger', 'sentencizer', 'parser', 'ner', 'entity_ruler']) 

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

    def test_split_into_sentences_str(self):
        """Test deepsegment to split a string into parts."""
        text_EDA = self._get_module('text_EDA')
        sents = text_EDA.split_into_sentences(TEST_UTTERANCES[3])
        self.assertEqual(len(sents), 2)

    def test_split_into_sentences_str_2(self):
        """Test deepsegment to split a string without punctuation into parts."""
        text_EDA = self._get_module('text_EDA')
        sents = text_EDA.split_into_sentences(TEST_UTTERANCES[4])
        self.assertEqual(len(sents), 2)

    def test_split_into_sentences(self):
        """Test SpaCy's ability to split a string into parts."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        sents = text_EDA.split_into_sentences(explorer.data['Raw Utterances'].iloc[3])
        self.assertEqual(len(sents), 2)

    def test_get_sentence_count_str(self):
        """Test getting the sentence count from a string."""
        text_EDA = self._get_module('text_EDA')
        cnt = text_EDA.get_sentence_count(TEST_UTTERANCES[3])
        self.assertEqual(cnt, 2)

    def test_get_sentence_count(self):
        """Test getting a sentence count from a SpaCy doc object."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        cnt = text_EDA.get_sentence_count(explorer.data['Raw Utterances'].iloc[3])
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

if __name__ == '__main__':
    unittest.main()