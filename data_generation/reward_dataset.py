import torch
from torch.utils.data import Dataset, DataLoader
import random
from data_generation.corruption import corrupt_tokens

class MidiBERTRewardDataset(Dataset):
    def __init__(self, maestro_tokens_list, tokenizer, block_size=512):
        """
        maestro_tokens_list : list of pieces (each piece is a list of ids).
        """
        self.data = maestro_tokens_list
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # take a perfect piece from the dataset
        perfect_tokens = self.data[idx]
        
        # randomly crop a slice to get the correct size (e.g., 512 tokens)
        if len(perfect_tokens) > self.block_size:
            start_idx = random.randint(0, len(perfect_tokens) - self.block_size)
            perfect_tokens = perfect_tokens[start_idx : start_idx + self.block_size]
            
        # randomly pick a severity for THIS specific batch (between 0.0 and 1.0)
        # 20% chance of having a perfect score (S=0.0) so MidiBERT knows what a real 10/10 is.
        if random.random() < 0.20:
            severity = 0.0
        else:
            severity = random.uniform(0.01, 1.0)
            
        # corruption
        corrupted_tokens, target_score = corrupt_tokens(
            perfect_tokens, 
            self.tokenizer, 
            severity=severity
        )
        
        # ensure the size is always block_size since corruption might have deleted or added tokens
        if len(corrupted_tokens) > self.block_size:
            corrupted_tokens = corrupted_tokens[:self.block_size]
        else:
            # padding with 0s if it's too short
            padding = [0] * (self.block_size - len(corrupted_tokens))
            corrupted_tokens.extend(padding)

        # return tensors
        x = torch.tensor(corrupted_tokens, dtype=torch.long)
        y = torch.tensor([target_score], dtype=torch.float)
        
        return x, y

