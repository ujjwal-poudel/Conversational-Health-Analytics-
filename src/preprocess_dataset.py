import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List

# Using good practice for optional import
try:
    # All transformers import goes here
    from transformers import AutoTokenizer
except ImportError:
    print("Warning: `transformers` library not found. Tokenization will not be available.")
    print("Please install it with `pip install transformers`")
    AutoTokenizer = None

class BasePreprocessor:
    """
    A class to handle loading, cleaning, and preparing text data for NLP models.
    """

    def clean_text(self, text_series: pd.Series) -> pd.Series:
        """
        Performs basic preprocessing on a pandas Series of text.
        - Converts text to lowercase.
        - Removes special characters.
        - Removes extra whitespace.
        
        Args:
            text_series (pd.Series): The raw text Series to preprocess.
        
        Returns:
            pd.Series: The processed text Series.
        """
        # Ensures all data is treated as a string, then convert to lowercase
        processed_series = text_series.astype(str).str.lower()
        
        # Remove angle brackets only
        processed_series = processed_series.str.replace(r'[<>]', '', regex=True)
        
        # Replaces multiple whitespace characters with a single space and strips leading/trailing spaces
        processed_series = processed_series.str.replace(r'\s+', ' ', regex=True).str.strip()
        
        print("Text cleaning complete.")
        return processed_series
    

    def load_and_preprocess(self, data_path: Path, text_column: str = 'text', label_column: str = 'label', id_column: str = 'participant_id',
        keep_id_column: bool = False) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Loads and preprocesses data. Now includes an option to keep an ID column.

        Args:
            data_path (Path): Path to the input CSV file.
            text_column (str): Name of the text column.
            label_column (str): Name of the label column.
            id_column (str): Name of the participant ID column.
            keep_id_column (bool): If True, keeps the id_column in the output X DataFrame.
                                   Defaults to False.

        Returns:
            A tuple of (X_df, y_df) or None if an error occurs.
        """
        if not data_path.is_file():
            print(f"Error: File not found at path: {data_path}")
            return None
        
        try:
            df = pd.read_csv(data_path)
            print(f"Successfully loaded {data_path}. Found {len(df)} records.")
        except Exception as e:
            print(f"Error: Failed to read the CSV file. Details: {e}")
            return None

        # Dynamically sets the required columns
        required_columns = [text_column, label_column]
        if keep_id_column:
            required_columns.append(id_column)

        if not all(col in df.columns for col in required_columns):
            print(f"Error: Not all required columns ({required_columns}) found.")
            print(f"Available columns are: {list(df.columns)}")
            return None

        df_processed = df[required_columns].copy()
        df_processed[text_column] = self.clean_text(df_processed[text_column])

        # Separates features (X) and labels (y)
        y = df_processed[[label_column]]
        
        feature_columns = [text_column]
        if keep_id_column:
            feature_columns.append(id_column)
        X = df_processed[feature_columns]

        print(f"Data prepared. X shape: {X.shape}, y shape: {y.shape}\n")
        return X, y
    
    def _create_text_chunks(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Private helper method to split a single text into overlapping chunks.
        It uses the 'self.tokenizer' provided by the specific subclass.
        """
        # This uses the tokenizer specific to the child class (DistilBert, Roberta, etc.)
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= chunk_size:
            return [text]

        step_size = chunk_size - overlap
        chunks = []
        for i in range(0, len(tokens) - overlap, step_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks
    
    def chunk_dataframe(self, X_df: pd.DataFrame, y_df: pd.DataFrame, 
                        chunk_size: int = 512, overlap: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Applies chunking to an entire dataframe. Expands the dataframe so each
        row represents a single chunk.
        """
        print(f"Starting chunking with chunk_size={chunk_size} and overlap={overlap}...")
        # Combines X and y to ensure labels stay aligned with their text
        df = X_df.join(y_df)
        
        new_rows = []
        for _, row in df.iterrows():
            original_text = row['text']
            # Use the private helper method for chunking
            text_chunks = self._create_text_chunks(original_text, chunk_size, overlap)
            
            for chunk in text_chunks:
                new_row = row.to_dict() # Copys all original columns (like participant_id)
                new_row['text'] = chunk # Replaces original text with the chunk
                new_rows.append(new_row)
        
        chunked_df = pd.DataFrame(new_rows)
        print(f"Chunking complete. Original docs: {len(df)}, Total chunks: {len(chunked_df)}")

        # Separate back into X and y
        y_chunked = chunked_df[['label']]
        X_chunked = chunked_df.drop('label', axis=1)

        return X_chunked, y_chunked

    def tokenize(self, text_series: pd.Series, max_length: int = 512):
        """Placeholder for the tokenization method."""
        raise NotImplementedError("This method should be implemented by a subclass.")
    
    
    
class DistilBertPreprocessor(BasePreprocessor):
    """Preprocessor for DistilBERT. Inherits chunking logic."""
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        super().__init__()
        if AutoTokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ImportError("`transformers` library is required.")

    def tokenize(self, text_series: pd.Series, max_length: int = 512) -> dict:
        print(f"Tokenizing for DistilBERT with max_length={max_length}...")
        # The tokenizer is now used for both chunking and final tokenization
        return self.tokenizer(
            text_series.tolist(), padding=True, truncation=True, 
            max_length=max_length, return_tensors='pt'
        )



class RobertaPreprocessor(BasePreprocessor):
    """Preprocessor for RoBERTa. Inherits chunking logic."""
    # ... (no changes needed here)
    def __init__(self, model_name: str = 'roberta-base'):
        super().__init__()
        if AutoTokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ImportError("`transformers` library is required.")
            
    def tokenize(self, text_series: pd.Series, max_length: int = 512) -> dict:
        print(f"Tokenizing for RoBERTa with max_length={max_length}...")
        return self.tokenizer(
            text_series.tolist(), padding=True, truncation=True, 
            max_length=max_length, return_tensors='pt'
        )
