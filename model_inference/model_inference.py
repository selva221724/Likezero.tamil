import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
import torch
import gdown

from dotenv import load_dotenv

# Load environment variables
load_dotenv() 


# Get the directory of the current script or notebook
current_directory = os.path.abspath(os.getcwd())
# Navigate to the desired directory 
parent_directory = os.path.dirname(current_directory)
os.chdir(parent_directory)
print(f"Current working directory: {os.getcwd()}")

from utils import DataProcessor

class NewsDataset(Dataset):
    """Custom dataset for news text classification.
    
    Args:
        texts (pd.Series): A pandas series containing news texts.
        labels (pd.Series): A pandas series containing news labels.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): A tokenizer object.
        max_length (int): Maximum length of the input sequence.
    """
    
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class InferenceData():
    """Class to perform inference on new data using a fine-tuned DistilBERT model.
    
    Attributes:
        model (transformers.modeling_distilbert.DistilBertForSequenceClassification): A fine-tuned DistilBERT model.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizerBase): A tokenizer object.
        data (pd.DataFrame): A pandas DataFrame containing the inference data.
        texts (list): A list of news texts.
        batch_size (int): The batch size for inference.
        model_name (str): The name of the DistilBERT model.
        fine_tuned_model_path (str): The path to the fine-tuned model."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.data = None
        self.texts = None
        self.batch_size = int(os.environ.get("BATCH_SIZE"))
        self.model_name = os.environ.get("MODEL_NAME")
        self.fine_tuned_model_path = os.environ.get("FINE_TUNED_MODEL_PATH")
        
        
    def load_model(self):
        """Load the fine-tuned DistilBERT model and tokenizer."""
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        
        # Check if the model is available inside the fine_tuned_model_path
        if not os.path.exists(self.fine_tuned_model_path):
            os.makedirs(self.fine_tuned_model_path)  # Create directory if it doesn't exist
        
        # check if the model is available inside the fine_tuned_model_path
        if not os.path.exists(self.fine_tuned_model_path + "/model.safetensors") and \
        not os.path.exists(self.fine_tuned_model_path + "/config.json"):
            print("Downloading the fine-tuned model...")
            config_file_id = "122F6JBh8o_N7K4fGemdNcjjt2c7XoxMD"
            model_file_id = "11wYbz4wkN7ox8vzy5POClIiHuOBsouX_"
            download_file_from_google_drive(model_file_id,os.path.join(os.getcwd(), self.fine_tuned_model_path +"/model.safetensors"))
            download_file_from_google_drive(config_file_id,os.path.join(os.getcwd(), self.fine_tuned_model_path +"/config.json"))
        
        self.model = DistilBertForSequenceClassification.from_pretrained(self.fine_tuned_model_path)
        
    def predict_on_new_data(self,texts):
        """Perform inference on new data."""
        self.texts = texts
        # Remove None samples
        inference_dataset = [sample for sample in self.texts if sample is not None]
        # Initialize DataLoader
        inference_dataloader = DataLoader(inference_dataset, batch_size=self.batch_size)
        predictions = []
        for batch in inference_dataloader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                batch_predictions = torch.argmax(outputs.logits, axis=1)
                predictions.extend(batch_predictions.cpu().numpy())
        return predictions

def download_file_from_google_drive(file_id, save_path):
    """Download a file from Google Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, save_path, quiet=False)


def main():
    # Load environment variables
    results_field = os.environ.get("RESULTS_FIELD")
    max_length = int(os.environ.get("MAX_LENGTH"))
    target_field = os.environ.get("TARGET_FIELD")
    results_path = os.environ.get("RESULTS_PATH")
    true_label = int(os.environ.get("TRUE_LABEL"))
    fake_label = int(os.environ.get("FAKE_LABEL"))
    ground_truth_available = os.environ.get("GROUND_TRUTH")
    
    
    # Convert save_model to boolean
    if ground_truth_available.lower() == 'true':
        ground_truth_available = True
        ground_truth_field = os.environ.get("GROUND_TRUTH_FIELD")
        print(f"Ground truth field available: {ground_truth_field}")
    else:
        ground_truth_available = False
        print("Ground truth field not available.")
    
    
    # Load and preprocess data
    data_processor = DataProcessor()
    data_processor.load_data()
    data_processor.preprocess_data()
    preprocessed_data = data_processor.get_preprocessed_data()
    
    # print the first few rows of the preprocessed data
    print("=============== Preprocessed data ====================")
    print(preprocessed_data.head())
    
    # Load the model and tokenizer
    inference = InferenceData()
    inference.load_model()
    
    # Create a dataset for inference
    predict_dataset = NewsDataset(preprocessed_data[target_field], inference.tokenizer, max_length=max_length)
    
    print(f"Prediction dataset size: {len(predict_dataset)}")

    # Perform inference
    inference_results = inference.predict_on_new_data(predict_dataset)
    print("Inference complete.")
    
    # Map the inference results to the corresponding labels
    preprocessed_data[results_field] = inference_results
    preprocessed_data[results_field] = preprocessed_data[results_field].map({true_label: 'Real', fake_label: 'Fake'}) 
    
    # Calculate accuracy
    if ground_truth_available:
        accuracy = accuracy_score(preprocessed_data[ground_truth_field], preprocessed_data[results_field])
        print(f"Accuracy on Inference Data: {accuracy}")
    
    # Save the results    
    preprocessed_data.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main()
