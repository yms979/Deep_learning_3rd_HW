import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import Shakespeare
from model import CharRNN, CharLSTM
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, trn_loader, device, criterion, optimizer, clip=1):
    model.train()
    trn_loss = 0
    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = model.init_hidden(inputs.size(0))
        hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(inputs, hidden)
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        trn_loss += loss.item()
    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            hidden = tuple(h.to(device) for h in hidden) if isinstance(hidden, tuple) else hidden.to(device)
            
            outputs, _ = model(inputs, hidden)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_file = 'shakespeare_train.txt'
    batch_size = 64
    hidden_size = 256
    num_layers = 2
    dropout = 0.5
    num_epochs = 100
    patience = 5 
    
    dataset = Shakespeare(input_file)
    trn_len = int(0.8 * len(dataset))
    val_len = len(dataset) - trn_len
    trn_set, val_set = random_split(dataset, [trn_len, val_len])
    
    trn_loader = DataLoader(trn_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    input_size = len(dataset.chars)
    output_size = input_size
    
    rnn_model = CharRNN(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    lstm_model = CharLSTM(input_size, hidden_size, output_size, num_layers, dropout).to(device)
    
    rnn_optimizer = torch.optim.Adam(rnn_model.parameters())
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    best_rnn_val_loss = float('inf')
    best_lstm_val_loss = float('inf')
    rnn_trn_losses = []
    lstm_trn_losses = []
    rnn_val_losses = []
    lstm_val_losses = []
    
    early_stopping_rnn = EarlyStopping(patience=patience, verbose=True)
    early_stopping_lstm = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in tqdm(range(num_epochs)):
        rnn_trn_loss = train(rnn_model, trn_loader, device, criterion, rnn_optimizer)
        rnn_val_loss = validate(rnn_model, val_loader, device, criterion)
        lstm_trn_loss = train(lstm_model, trn_loader, device, criterion, lstm_optimizer)
        lstm_val_loss = validate(lstm_model, val_loader, device, criterion)
        
        rnn_trn_losses.append(rnn_trn_loss)
        lstm_trn_losses.append(lstm_trn_loss)
        rnn_val_losses.append(rnn_val_loss)
        lstm_val_losses.append(lstm_val_loss)
        
        print(f'Epoch: {epoch+1}/{num_epochs}, RNN - Train Loss: {rnn_trn_loss:.4f}, Val Loss: {rnn_val_loss:.4f}')
        print(f'Epoch: {epoch+1}/{num_epochs}, LSTM - Train Loss: {lstm_trn_loss:.4f}, Val Loss: {lstm_val_loss:.4f}')
        
        early_stopping_rnn(rnn_val_loss, rnn_model)
        if rnn_val_loss < best_rnn_val_loss:
            best_rnn_val_loss = rnn_val_loss
            torch.save(rnn_model.state_dict(), 'rnn_model.pth')
        
        early_stopping_lstm(lstm_val_loss, lstm_model)
        if lstm_val_loss < best_lstm_val_loss:
            best_lstm_val_loss = lstm_val_loss
            torch.save(lstm_model.state_dict(), 'lstm_model.pth')
        
        if early_stopping_rnn.early_stop or early_stopping_lstm.early_stop:
            print("Early stopping")
            break
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(rnn_trn_losses, label='RNN Training Loss')
    plt.plot(rnn_val_losses, label='RNN Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('RNN Training and Validation Loss Curves')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(lstm_trn_losses, label='LSTM Training Loss')
    plt.plot(lstm_val_losses, label='LSTM Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training and Validation Loss Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

if __name__ == '__main__':
    main()