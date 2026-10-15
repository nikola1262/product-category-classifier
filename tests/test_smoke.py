"""Brzi testovi: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predict_category import predict_title
from src.data_utils import clean_data


class ProductClassifierTests(unittest.TestCase):
    def test_cleaning_aliases_and_empty_rows(self):
        raw = pd.DataFrame({
            "Product Title": ["Model one", " Model one ", "Model two", "Model three", ""],
            " Category Label ": ["CPU", "CPUs", "Mobile Phone", "fridge", "TVs"],
        })
        # Dve instance istog modela postaju jedna, a aliasi se normalizuju.
        cleaned, stats = clean_data(raw)
        self.assertEqual(len(cleaned), 3)
        self.assertEqual(stats["duplicate_title_rows_removed"], 1)
        self.assertIn("CPUs", set(cleaned.category))
        self.assertIn("Mobile Phones", set(cleaned.category))
        self.assertIn("Fridges", set(cleaned.category))

    def test_conflicting_labels_removed(self):
        raw = pd.DataFrame({
            "Product Title": ["Same item", "same item", "Other item", "Third item"],
            "Category Label": ["TVs", "CPUs", "TVs", "CPUs"],
        })
        cleaned, stats = clean_data(raw)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(stats["ambiguous_title_groups"], 1)

    def test_saved_model_on_examples(self):
        path = ROOT / "models" / "product_category_model.pkl"
        self.assertTrue(path.exists(), "Pre testa pokreni: python train_model.py")
        model = joblib.load(path)
        cases = [
            ("iphone 7 32gb gold,4,3,Apple iPhone 7 32GB", "Mobile Phones"),
            ("olympus e m10 mark iii geh use silber", "Digital Cameras"),
            ("kenwood k20mss15 solo", "Microwaves"),
            ("bosch wap28390gb 8kg 1400 spin", "Washing Machines"),
            ("bosch serie 4 kgv39vl31g", "Fridge Freezers"),
            ("smeg sbs8004po", "Fridge Freezers"),
        ]
        for title, expected in cases:
            with self.subTest(title=title):
                self.assertEqual(predict_title(model, title), expected)


if __name__ == "__main__":
    unittest.main()
