from src.evaluate.metrics import compute_exact_match

def test_exact_match():
    # Test that identical strings return 1.0
    assert compute_exact_match("Hello World", "hello world") == 1.0
    
    # Test that different strings return 0.0
    assert compute_exact_match("Hello World", "Goodbye") == 0.0
