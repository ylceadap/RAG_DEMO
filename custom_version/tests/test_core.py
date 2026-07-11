import unittest

from ingest import split_text


class CoreFunctionTests(unittest.TestCase):
    def test_split_text_keeps_all_content(self):
        text = "a" * 1500
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0], "a" * 700)
        self.assertEqual(chunks[-1][-1], "a")

if __name__ == "__main__":
    unittest.main()
