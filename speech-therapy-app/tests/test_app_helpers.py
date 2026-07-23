import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import app


class TestSpeechHelpers(unittest.TestCase):
    def test_analyze_filler_words_counts_common_fillers(self):
        result = app.analyze_filler_words("um uh like um", "English")
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["counts"]["um"], 2)
        self.assertEqual(result["counts"]["uh"], 1)
        self.assertEqual(result["counts"]["like"], 1)

    def test_analyze_filler_words_falls_back_for_unknown_language(self):
        result = app.analyze_filler_words("uh um", "unknown")
        self.assertEqual(result["total"], 2)
        self.assertGreaterEqual(result["counts"]["uh"], 1)

    def test_analyze_transcript_quality_flags_unclear_tokens(self):
        result = app.analyze_transcript_quality("n ar hello hlo", "English")
        self.assertIn("n", result["unclear_tokens"])
        self.assertIn("hlo", result["unclear_tokens"])
        self.assertIn("unclear", result["message"].lower())


if __name__ == "__main__":
    unittest.main()
