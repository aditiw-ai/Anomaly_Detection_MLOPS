from app.services.ab_testing_service import ABTestingService


def test_parse_binary_label_accepts_numeric_strings():
    assert ABTestingService._parse_binary_label("1.0") == 1
    assert ABTestingService._parse_binary_label("0.0") == 0
    assert ABTestingService._parse_binary_label(" 1 ") == 1
    assert ABTestingService._parse_binary_label("0") == 0
    # Simulate numpy scalar compatibility without importing numpy in service code.
    class NpLike:
        dtype = "int64"
        def __init__(self, v):
            self.v = v
        def item(self):
            return self.v
        def __float__(self):
            return float(self.v)
    assert ABTestingService._parse_binary_label(NpLike(1)) == 1


def test_infer_label_key_detects_common_class_column():
    rows = [
        {"Amount": 100.0, "Class": 0},
        {"Amount": 22.5, "Class": 1},
        {"Amount": 40.1, "Class": 0},
        {"Amount": 8.7, "Class": 1},
    ]
    inferred = ABTestingService._infer_label_key(rows)
    assert inferred == "Class"


def test_extract_actual_label_with_class_and_y_variants():
    assert ABTestingService._extract_actual_label({"Class": "1.0"}) == 1
    assert ABTestingService._extract_actual_label({"y": "0.0"}) == 0
    assert ABTestingService._extract_actual_label({"response": "yes"}) == 1
    # Space and hyphen variants should also resolve.
    assert ABTestingService._extract_actual_label({"is fraud": 1}) == 1
    assert ABTestingService._extract_actual_label({"is-fraud": 0}) == 0
