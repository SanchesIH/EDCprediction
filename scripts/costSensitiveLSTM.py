import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
import optuna
from tqdm import tqdm
from generateDataset import class_weights, ProgesteroneDataset
from sklearn.metrics import confusion_matrix

# Initialize ChemBERTa tokenizer
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-10M-MLM')
base_model = AutoModelForMaskedLM.from_pretrained('DeepChem/ChemBERTa-10M-MLM')

# Define the LSTM model with dropout and transfer learning from ChemBERTa
class LSTMModel(nn.Module):
    def __init__(self, base_model, hidden_dim, output_dim, dropout_rate, num_layers=2, num_heads=4):
        super(LSTMModel, self).__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(base_model.config.hidden_size, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, input_ids, attention_mask):
        # Get hidden states from the base model (ChemBERTa)
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        # LSTM layer
        h0 = torch.zeros(self.num_layers, hidden_states.size(0), self.hidden_dim).to(hidden_states.device)
        c0 = torch.zeros(self.num_layers, hidden_states.size(0), self.hidden_dim).to(hidden_states.device)
        out_lstm, _ = self.lstm(hidden_states, (h0, c0))
        
        # Multi-Head Attention layer
        out_attn, _ = self.multihead_attn(out_lstm.transpose(0, 1), out_lstm.transpose(0, 1), out_lstm.transpose(0, 1))
        out_attn = out_attn.transpose(0, 1)
        
        # Final output layer
        out = self.fc(self.dropout(out_attn[:, -1, :]))
        
        return out

# Load the saved datasets
with open(r'C:\Users\igorh\Documents\Progesterone\data\customDataset\train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

with open(r'C:\Users\igorh\Documents\Progesterone\data\customDataset\test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
    num_epochs = trial.suggest_int('num_epochs', 3, 10)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    hidden_dim = 128
    output_dim = 2
    model = LSTMModel(base_model, hidden_dim, output_dim, dropout_rate=dropout_rate).to(device)
    
    # Define loss function with class weights
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Validation loop
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate sensitivity and specificity
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Calculate balanced accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    return balanced_accuracy

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Save study trials to CSV
study.trials_dataframe().to_csv(r'C:\Users\igorh\Documents\Progesterone\optunaStudy\optuna_trials_costSensitiveLSTM_agonist.csv', index=False)

# Print the best hyperparameters
print(f"Best hyperparameters: {study.best_params}")
print(f"Best balanced accuracy: {study.best_value}")