import unittest

from rag import calculate_math, extract_math_expression


class LangChainCoreTests(unittest.TestCase):
    def test_simple_math(self):
        self.assertEqual(extract_math_expression("What is 1+1?"), "1+1")
        self.assertEqual(calculate_math("(2+3)*4"), 20)

    def test_math_rejects_function_calls(self):
        with self.assertRaises(ValueError):
            calculate_math("abs(-1)")


if __name__ == "__main__":
    unittest.main()
