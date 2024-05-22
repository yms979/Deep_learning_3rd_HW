import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    def __init__(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        
        self.data = [self.char_to_idx[char] for char in text]
        self.seq_len = 30
        
        self.sequences = []
        self.targets = []
        for i in range(len(self.data) - self.seq_len):
            self.sequences.append(self.data[i:i+self.seq_len])
            self.targets.append(self.data[i+1:i+self.seq_len+1])
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_seq = self.sequences[idx]
        target_seq = self.targets[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)