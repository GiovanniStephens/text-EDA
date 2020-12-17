import unittest

TEST_UTTERANCES = (
    'This is a test sentence.'
    , 'This is a similar test phrase.'
    , 'Nothing to do with the others.'
    , 'This is a multi-sentence phrase. I want to know if I can split them.'
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

    def test_split_into_sentences_str(self):
        """Test SpaCy's ability to split a string into parts."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        sents = text_EDA.split_into_sentences(explorer.data['Raw Utterances'].iloc[3])
        self.assertEqual(len(sents), 2)

if __name__ == '__main__':
    unittest.main()