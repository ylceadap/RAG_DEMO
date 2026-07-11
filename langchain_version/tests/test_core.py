import unittest

from ingest import CHUNK_OVERLAP, CHUNK_SIZE, get_text_splitter


class LangChainCoreTests(unittest.TestCase):
    def test_text_splitter_uses_project_settings(self):
        splitter = get_text_splitter()
        self.assertEqual(splitter._chunk_size, CHUNK_SIZE)
        self.assertEqual(splitter._chunk_overlap, CHUNK_OVERLAP)


if __name__ == "__main__":
    unittest.main()
