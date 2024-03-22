import os
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
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
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]

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
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class WeightedTrainer(Trainer):
    """Custom trainer class to handle class weights in the loss function.
    
    Args:
        class_weight (torch.Tensor): A tensor containing class weights.
        """
        
    def __init__(self, class_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight = class_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weight.to(outputs.logits.device, dtype=torch.float32))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_model(model, train_dataset, test_dataset, class_weights, training_args):
    trainer = WeightedTrainer(
        class_weight=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=None
    )
    trainer.train()

    return trainer

def evaluate_model(trained_model, test_dataset):
    predictions = trained_model.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = test_dataset.labels.to_numpy()

    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')

    return accuracy, precision, recall, f1_score

def main():
    
    target_field = os.environ.get("TARGET_FIELD")
    label_field = os.environ.get("LABEL_FIELD")
    output_dir = os.environ.get("OUTPUT_DIR")
    model_name = os.environ.get("MODEL_NAME")
    per_device_train_batch_size = int(os.environ.get("BATCH_SIZE"))
    per_device_eval_batch_size = int(os.environ.get("BATCH_SIZE"))
    num_train_epochs = int(os.environ.get("EPOCHS"))
    evaluation_strategy = os.environ.get("EVALUATION_STRATEGY")
    save_strategy = os.environ.get("SAVE_STRATEGY")
    logging_dir = os.environ.get("LOGGING_DIR")
    logging_steps = int(os.environ.get("LOGGING_STEPS"))
    save_steps = int(os.environ.get("SAVE_STEPS"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY"))
    max_length = int(os.environ.get("MAX_LENGTH"))
    train_size = float(os.environ.get("TRAIN_SIZE"))
    random_state = int(os.environ.get("RANDOM_STATE"))
    save_model = os.environ.get("SAVE_MODEL")
    save_model_path = os.environ.get("SAVE_MODEL_PATH")
    
    # Convert save_model to boolean
    if save_model.lower() == 'true':
        save_model = True
    else:
        save_model = False
    
    
    # Load and preprocess data
    data_processor = DataProcessor()
    data_processor.load_data()
    data_processor.preprocess_data()
    preprocessed_data = data_processor.get_preprocessed_data()
    
    # print the first few rows of the preprocessed data
    print("=============== Preprocessed data ====================")
    print(preprocessed_data.head())
    
    # Split data into training and testing sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        preprocessed_data[target_field], preprocessed_data[label_field], test_size=1-train_size, random_state=random_state)
    
    print(f"Training set size: {len(train_texts)}")
    print(f"Testing set size: {len(test_texts)}")

    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Calculate class weights for imbalanced dataset
    class_counts = preprocessed_data[label_field].value_counts()
    class_weights = torch.tensor([class_counts[0] / class_counts[1], 1.0])

    # Create datasets
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, max_length)
    test_dataset = NewsDataset(test_texts, test_labels, tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        save_steps=save_steps,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_first_step=True,
        load_best_model_at_end=True,
        greater_is_better=True
    )

    # Train model
    trained_model = train_model(model, train_dataset, test_dataset, class_weights, training_args)
    
    # Save model
    if save_model:
        trained_model.model.save_pretrained(save_model_path)

    # Evaluate model
    accuracy, precision, recall, f1_score = evaluate_model(trained_model, test_dataset)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1_score}")

if __name__ == "__main__":
    main()
