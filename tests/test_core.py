import unittest

from ingest import split_text
from rag import calculate_math, extract_math_expression


class CoreFunctionTests(unittest.TestCase):
    def test_split_text_keeps_all_content(self):
        text = "a" * 1500
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0], "a" * 700)
        self.assertEqual(chunks[-1][-1], "a")

    def test_simple_math_detection(self):
        self.assertEqual(extract_math_expression("What is 1+1?"), "1+1")
        self.assertEqual(calculate_math("(2+3)*4"), 20)

    def test_math_rejects_function_calls(self):
        with self.assertRaises(ValueError):
            calculate_math("abs(-1)")


if __name__ == "__main__":
    unittest.main()
