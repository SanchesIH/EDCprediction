import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


# Load Dataset and split data
dataset_path = r"C:\Users\igorh\Documents\Progesterone\data\curated_binary_agonist.csv"
dataset = pd.read_csv(dataset_path)
X_train, X_test, y_train, y_test = train_test_split(dataset['SMILES'], dataset['Outcome'], test_size=0.2, random_state=42, stratify=dataset['Outcome'])

# concat train df
train_df = pd.DataFrame({'SMILES': X_train, 'Outcome': y_train}).reset_index(drop=True)

# concat test df
test_df = pd.DataFrame({'SMILES': X_test, 'Outcome': y_test}).reset_index(drop=True)

# Initialize ChemBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-10M-MLM')

# Encode SMILES using ChemBERTa tokenizer
train_encoded_inputs = tokenizer(train_df['SMILES'].tolist(), padding=True, truncation=True, return_tensors="pt")
test_encoded_inputs = tokenizer(test_df['SMILES'].tolist(), padding=True, truncation=True, return_tensors="pt")

# Encode outcomes
label_encoder = LabelEncoder()
train_encoded_outcomes = label_encoder.fit_transform(train_df['Outcome'])
test_encoded_outcomes = label_encoder.fit_transform(test_df['Outcome'])


# Calculate class weights
class_counts = torch.tensor([(train_encoded_outcomes == 0).sum(), (train_encoded_outcomes == 1).sum()])
class_weights = 1. / class_counts.float()

# Create a custom dataset
class ProgesteroneDataset():
    def __init__(self, encodings, outcomes):
        self.encodings = encodings
        self.outcomes = outcomes

    def __len__(self):
        return len(self.outcomes)

    def __getitem__(self, idx):
        item = {key: val[idx].detach().clone() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.outcomes[idx])
        return item
    
train_dataset = ProgesteroneDataset(train_encoded_inputs, train_encoded_outcomes)
test_dataset = ProgesteroneDataset(test_encoded_inputs, test_encoded_outcomes)

# Save the datasets into pkl files
with open(r'C:\Users\igorh\Documents\Progesterone\data\customDataset\train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)

with open(r'C:\Users\igorh\Documents\Progesterone\data\customDataset\test_dataset.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)

