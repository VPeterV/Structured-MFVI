import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CharLSTM(nn.Module):
    r"""
    CharLSTM aims to generate character-level embeddings for tokens.
    It summarizes the information of characters in each token to an embedding using a LSTM layer.

    Args:
        n_char (int):
            The number of characters.
        n_embed (int):
            The size of each embedding vector as input to LSTM.
        n_hidden (int):
            The size of each LSTM hidden state.
        n_out (int):
            The size of each output vector. Default: 0.
            If 0, equals to the size of hidden states.
        pad_index (int):
            The index of the padding token in the vocabulary. Default: 0.
        dropout (float):
            The dropout ratio of CharLSTM hidden states. Default: 0.
    """

    def __init__(self, n_chars, n_embed, n_hidden, n_out=0, pad_index=0, dropout=0):
        super().__init__()

        self.n_chars = n_chars
        self.n_embed = n_embed
        self.n_hidden = n_hidden
        self.n_out = n_out or n_hidden
        self.pad_index = pad_index

        self.embed = nn.Embedding(num_embeddings=n_chars, embedding_dim=n_embed)
        self.lstm = nn.LSTM(input_size=n_embed, hidden_size=n_hidden//2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=dropout)
        self.projection = nn.Linear(in_features=n_hidden, out_features=self.n_out) if n_hidden != self.n_out else nn.Identity()

    def __repr__(self):
        s = f"{self.n_chars}, {self.n_embed}"
        if self.n_hidden != self.n_out:
            s += f", n_hidden={self.n_hidden}"
        s += f", n_out={self.n_out}, pad_index={self.pad_index}"
        if self.dropout.p != 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor): ``[batch_size, seq_len, fix_len]``.
                Characters of all tokens.
                Each token holds no more than `fix_len` characters, and the excess is cut off directly.
        Returns:
            ~torch.Tensor:
                The embeddings of shape ``[batch_size, seq_len, n_out]`` derived from the characters.
        """

        # [batch_size, seq_len, fix_len]
        mask = x.ne(self.pad_index)
        # [batch_size, seq_len]
        lens = mask.sum(-1)
        char_mask = lens.gt(0)

        # [n, fix_len, n_embed]
        x = self.embed(x[char_mask])
        x = pack_padded_sequence(x, lens[char_mask].tolist(), True, False)
        x, (h, _) = self.lstm(x)
        # [n, fix_len, n_hidden]
        h = self.dropout(torch.cat(torch.unbind(h), -1))
        # [batch_size, seq_len, n_out]
        embed = h.new_zeros(*lens.shape, self.n_out).masked_scatter_(char_mask.unsqueeze(-1), self.projection(h))

        return embed