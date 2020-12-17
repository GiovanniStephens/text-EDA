import unittest

TEST_UTTERANCES = (
    'This is a test sentence.'
    , 'This is a similar test phrase.'
    , 'Nothing to do with the others.'
)

class test_embeddings(unittest.TestCase):

    def _get_module(self, module_name):
        import importlib
        return importlib.import_module(module_name)

    def test_get_word_count(self):
        """Tests that we're getting the right word count."""
        text_EDA = self._get_module('text_EDA')
        self.assertEqual(text_EDA.get_word_count(TEST_UTTERANCES[0]), 5)

    def test_load_nlp_pipe(self):
        """Tests that the nlp pipe has been set up correctly."""
        text_EDA = self._get_module('text_EDA')
        explorer = text_EDA.text_EDA(TEST_UTTERANCES)
        self.assertEqual(explorer.nlp.pipe_names, \
             ['tagger', 'sentencizer', 'parser', 'ner', 'entity_ruler']) 

if __name__ == '__main__':
    unittest.main()