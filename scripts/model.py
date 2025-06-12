import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_mat, hidden_size, num_classes, pad_idx, bidirectional=False, dropout_p=0.2, emb_dropout_p=0.1):
        super(LSTMClassifier, self).__init__()

        vocab_size, emb_dim = embed_mat.shape
        embed_mat = torch.from_numpy(embed_mat)

        self.embedding = nn.Embedding.from_pretrained(
            embeddings=embed_mat, 
            freeze=True, 
            padding_idx=pad_idx)

        
        # dropout embeddings

        self.emb_dropout = nn.Dropout(embed_dropout_p)
            

        self.lstm = nn.LSTM(emb_dim, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths=None):
        x = self.embedding(x) # (B, T) -> (B, T, emb_dim)

        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # print(packed_input.data.shape)
        # → torch.Size([sum(lengths), D])

        packed_out, (hn, cn) = self.lstm(packed_input)
        # hn.shape == (D, B, H)

        last_hidden_state = hn[-1] # for whole description
        dropped = self.dropout(last_hidden_state)  # → (B, H)
        output = self.fc(dropped)
        return output


if __name__ == "__main__":
    cl = LSTMClassifier()
