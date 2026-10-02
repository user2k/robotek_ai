import unittest
from train import training_size


class CurriculumTests(unittest.TestCase):
    def test_curriculum_advances_one_size_only_at_full_success(self):
        self.assertEqual(training_size(5, 5, 1), (7, 7))
        self.assertEqual(training_size(7, 7, 0.95), (7, 7))
        self.assertEqual(training_size(13, 13, 1), (15, 15))
        self.assertEqual(training_size(15, 15, 1), (15, 15))

    def test_promotion_requires_twenty_wins_on_current_size(self):
        from train import advance_curriculum
        stage = {"width": 5, "height": 5, "wins": []}
        advance_curriculum(stage, False)
        for _ in range(19):
            advance_curriculum(stage, True)
        self.assertEqual(stage['width'], 5)
        advance_curriculum(stage, True)
        self.assertEqual(stage, {"width": 7, "height": 7, "wins": []})
        advance_curriculum(stage, True)
        self.assertEqual(stage['width'], 7)


if __name__ == "__main__":
    unittest.main()
