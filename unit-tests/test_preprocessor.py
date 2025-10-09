import pytest
import pandas as pd
import torch
from pathlib import Path

from src.preprocess_dataset import DistilBertPreprocessor

@pytest.fixture
def distilbert_preprocessor():
    """Returns an instance of the DistilBertPreprocessor."""
    return DistilBertPreprocessor()

@pytest.fixture
def temp_csv_file(tmp_path):
    """Creates a temporary CSV file for testing file loading."""
    # tmp_path is a built-in pytest fixture that provides a temporary directory
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "test_data.csv"
    
    data = {
        'participant_id': [101, 102],
        'text': ["Some sample text here.", "Another document for &* (hello)  testing.  "],
        'label': [1, 0]
    }
    df = pd.DataFrame(data)
    df.to_csv(p, index=False)
    return p

# Tests for the BasePreprocessor logic
def test_clean_text(distilbert_preprocessor):
    """Tests if text is correctly lowercased and cleaned of punctuation."""
    raw_text = pd.Series(["  Hello World!! This is Test #1.  "])
    cleaned = distilbert_preprocessor.clean_text(raw_text)
    
    expected = "hello world this is test 1"
    assert cleaned.iloc[0] == expected

def test_load_and_preprocess_success(distilbert_preprocessor, temp_csv_file):
    """Tests successful loading and preparation of data."""
    X, y = distilbert_preprocessor.load_and_preprocess(temp_csv_file)
    
    assert X is not None
    assert y is not None
    assert X.shape == (2, 1)
    assert y.shape == (2, 1)
    assert 'text' in X.columns

def test_load_and_preprocess_with_id(distilbert_preprocessor, temp_csv_file):
    """Tests that participant_id is kept when the flag is True."""
    X, y = distilbert_preprocessor.load_and_preprocess(temp_csv_file, keep_id_column=True)

    assert 'participant_id' in X.columns
    assert X.shape == (2, 2)

def test_load_nonexistent_file(distilbert_preprocessor):
    """Tests that the function returns None for a file that doesn't exist."""
    result = distilbert_preprocessor.load_and_preprocess(Path("nonexistent/file.csv"))
    assert result is None

# Tests for Chunking and Tokenization
def test_chunking_on_long_text(distilbert_preprocessor):
    """Tests that long text is split into multiple chunks."""
    # Creates a long text (repeats a phrase 50 times)
    long_text = "this is a test sentence for chunking. " * 50
    X_df = pd.DataFrame({'participant_id': [1], 'text': [long_text]})
    y_df = pd.DataFrame({'label': [1]})
    
    # Uses a small chunk size for easy testing
    X_chunked, y_chunked = distilbert_preprocessor.chunk_dataframe(X_df, y_df, chunk_size=30, overlap=5)

    assert len(X_chunked) > 1 # Should create more than one chunk
    assert X_chunked['participant_id'].iloc[0] == 1
    assert y_chunked['label'].iloc[0] == 1
    assert len(y_chunked) == len(X_chunked) # Labels should match chunk count

def test_chunking_on_short_text(distilbert_preprocessor):
    """Tests that short text is not chunked."""
    short_text = "this is a short text."
    X_df = pd.DataFrame({'text': [short_text]})
    y_df = pd.DataFrame({'label': [0]})

    X_chunked, y_chunked = distilbert_preprocessor.chunk_dataframe(X_df, y_df)

    assert len(X_chunked) == 1 # Should only result in one chunk
    assert X_chunked['text'].iloc[0] == short_text

# In unit-tests/test_preprocessor.py

def test_distilbert_tokenize(distilbert_preprocessor):
    """
    Tests that tokenization returns the correct structure with torch tensors.
    """
    text_series = pd.Series(["hello world"])
    encodings = distilbert_preprocessor.tokenize(text_series)

    # Instead of checking if it's a dict, checks for what we really need:
    # Does it contain the keys we expect?
    assert 'input_ids' in encodings
    assert 'attention_mask' in encodings

    # Are the values the correct type (PyTorch Tensors)?
    assert isinstance(encodings['input_ids'], torch.Tensor)
    assert isinstance(encodings['attention_mask'], torch.Tensor)

    # Does the output shape make sense (1 sentence processed)?
    assert encodings['input_ids'].shape[0] == 1