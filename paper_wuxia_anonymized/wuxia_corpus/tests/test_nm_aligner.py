import unittest

import numpy as np

from src.alignment.nm_aligner import align_segments_nm


class MappingEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, *, normalize_embeddings):
        return np.asarray([self.vectors[text] for text in texts], dtype=float)


class NMAlignerTests(unittest.TestCase):
    def test_selects_a_many_to_many_transition(self):
        encoder = MappingEncoder({
            "s1": [1.0, 0.0], "s2": [1.0, 0.0],
            "t1": [-1.0, 0.0], "t2": [-1.0, 0.0],
            "s1 s2": [0.0, 1.0], "t1 t2": [0.0, 1.0],
        })
        aligned, scores, statistics = align_segments_nm(
            ["s1", "s2"], ["t1", "t2"], encoder,
            max_source_group=2, max_target_group=2,
        )
        self.assertEqual(aligned, [("s1 s2", "t1 t2")])
        self.assertEqual(scores, [1.0])
        self.assertEqual(statistics["2-2"], 1)

    def test_empty_input_is_accounted_for_as_skips(self):
        encoder = MappingEncoder({"target": [1.0, 0.0]})
        aligned, scores, statistics = align_segments_nm([], ["target"], encoder)
        self.assertEqual(aligned, [])
        self.assertEqual(scores, [])
        self.assertEqual(statistics["skip_en"], 1)


if __name__ == "__main__":
    unittest.main()
