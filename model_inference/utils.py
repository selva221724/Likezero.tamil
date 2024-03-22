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
        self.target_field = os.environ.get("TARGET_FIELD")
        self.predict_data_path = os.environ.get("PREDICT_DATA_PATH")
        self.label_field = os.environ.get("LABEL_FIELD")
        self.predict_data = None

    def load_data(self):
        """ Load the prediction data from the specified path.
        """
        self.predict_data = pd.read_csv(self.predict_data_path)
        print("Data loaded successfully.")

    def preprocess_data(self):
        """Preprocess the prediction data before inference."""
        if self.predict_data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        # Drop rows with missing values in the target field
        print(f"Number of rows with missing values getting dropped: {self.predict_data[self.target_field].isnull().sum()}")
        self.predict_data = self.predict_data.dropna(subset=[self.target_field])

        # Reset index
        self.predict_data = self.predict_data.reset_index(drop=True)
        
        # Replace text with 'nan' with empty string
        self.predict_data[self.target_field] = self.predict_data[self.target_field].replace('nan', '')
        print(f"Number of rows with 'nan' text: {self.predict_data[self.target_field].str.contains('nan').sum()}")
        
        # Lowercasing the text for pandas series wit apply
        self.predict_data[self.target_field] = self.predict_data[self.target_field].str.lower()
        
        # Remove punctuation
        self.predict_data[self.target_field] = self.predict_data[self.target_field].apply(lambda x: re.sub(r'[^\w\s]', '', x))
        
        # Remove extra whitespaces
        self.predict_data[self.target_field] = self.predict_data[self.target_field].apply(lambda x: re.sub(r'\s+', ' ', x))
        
        # Remove special characters and non-alphanumeric characters
        self.predict_data[self.target_field] = self.predict_data[self.target_field].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', ' ', x))
        
        # Remove stopwords
        self.predict_data[self.target_field] = self.predict_data[self.target_field].apply(remove_stopwords)
        
        print("Data preprocessing complete.")

    def get_preprocessed_data(self):
        """Get the preprocessed prediction data."""
        if self.predict_data is None:
            raise ValueError("Data not loaded. Please call load_data() and preprocess_data() first.")
        return self.predict_data
