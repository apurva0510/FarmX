import unittest

from cost import crop


class CropYieldTests(unittest.TestCase):
    def test_known_crop_returns_positive_yield(self):
        self.assertGreater(crop(100, 12, 1000, "maize"), 0)

    def test_unknown_crop_raises_value_error(self):
        with self.assertRaises(ValueError):
            crop(100, 12, 1000, "unknown")

    def test_zero_harvested_area_raises_value_error(self):
        with self.assertRaises(ValueError):
            crop(100, 12, 0, "maize")


if __name__ == "__main__":
    unittest.main()
