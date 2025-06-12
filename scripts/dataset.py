import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List


class MovieDataset(Dataset):
    def __init__(self, df, vocab, max_len=None):
        self.seqs = df.tokens.tolist()   
        self.labels = df.genre_id.tolist()
        self.vocab = vocab
        self.max_len = max_len          
        self.pad_id, = vocab.encode(["<pad>"])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = self.seqs[idx]
        if self.max_len is not None:
            toks = toks[: self.max_len]       
        ids  = self.vocab.encode(toks)    
        return torch.tensor(ids), torch.tensor(self.labels[idx])


def collate_batch(batch, pad_id=0):
    # unpacking indices and labels
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs]) 
    
    # ensures pad_id is created on the same device
    padded  = pad_sequence(seqs, batch_first=True, padding_value=pad_id)

    labels  = torch.tensor(labels)            
    return padded, lengths, labels


