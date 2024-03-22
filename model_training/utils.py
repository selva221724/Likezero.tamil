import os
import pandas as pd
from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)
    
    
class DataProcessor:
    def __init__(self):
        load_dotenv()
        self.fake_label = int(os.environ.get("FAKE_LABEL"))
        self.true_label = int(os.environ.get("TRUE_LABEL"))
        self.target_field = os.environ.get("TARGET_FIELD")
        self.fake_data_path = os.environ.get("FAKE_DATA_PATH")
        self.true_data_path = os.environ.get("TRUE_DATA_PATH")
        self.label_field = os.environ.get("LABEL_FIELD")
        self.combined_data = None

    def load_data(self):
        """Load fake and true news data and preprocess it for training."""
        fake_data = pd.read_csv(self.fake_data_path)
        true_data = pd.read_csv(self.true_data_path)

        fake_data[self.label_field] = self.fake_label  # Fake news label
        true_data[self.label_field] = self.true_label  # Real news label

        self.combined_data = pd.concat([fake_data, true_data])
        print("Data loaded successfully.")

    def preprocess_data(self):
        """Preprocess the combined fake and true news data for training."""
        if self.combined_data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Drop rows with missing values in the target field
        print(f"Number of rows with missing values getting dropped: {self.combined_data[self.target_field].isnull().sum()}")
        self.combined_data = self.combined_data.dropna(subset=[self.target_field])

        # Reset index
        self.combined_data = self.combined_data.reset_index(drop=True)
        
        # Replace text with 'nan' with empty string
        self.combined_data[self.target_field] = self.combined_data[self.target_field].replace('nan', '')
        print(f"Number of rows with 'nan' text: {self.combined_data[self.target_field].str.contains('nan').sum()}")
        
        # Lowercasing the text for pandas series wit apply
        self.combined_data[self.target_field] = self.combined_data[self.target_field].str.lower()
        
        # Remove punctuation
        self.combined_data[self.target_field] = self.combined_data[self.target_field].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # Remove extra whitespaces
        self.combined_data[self.target_field] = self.combined_data[self.target_field].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        # Remove special characters and non-alphanumeric characters
        self.combined_data[self.target_field] = self.combined_data[self.target_field].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x))
        
        # Remove stopwords
        self.combined_data[self.target_field] = self.combined_data[self.target_field].apply(remove_stopwords)
        
        print("Data preprocessing complete.")

    def get_preprocessed_data(self):
        """Get the preprocessed fake and true news data."""
        if self.combined_data is None:
            raise ValueError("Data not loaded. Please call load_data() and preprocess_data() first.")
        return self.combined_data
