import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, draft_length, embedding_dim, num_lstm_units, num_lstm_layers, dataset, device):
        super().__init__()
        self.draft_length = draft_length

        self.embedding = nn.Embedding(
            num_embeddings=draft_length+12,
            embedding_dim=embedding_dim
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=num_lstm_units,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        self.h2o = nn.Linear(num_lstm_units, draft_length)
        self.device = device
        self.dataset = dataset
        self.to(device)

    def forward(self, input, hidden=None):
        """
        Predict the next token's logits given an input token and a hidden state.
        :param input [torch.tensor]: The input token tensor with shape
            (batch_size, 1), where batch_size is the number of inputs to process
            in parallel.
        :param hidden [(torch.tensor, torch.tensor)]: The hidden state, or None if
            it's the first token.
        :return [(torch.tensor, (torch.tensor, torch.tensor))]: A tuple consisting of
            the logits for the next token, of shape (batch_size, num_tokens), and
            the next hidden state.
        """
        embeddings = self.embedding(input)
        if hidden is None:
            lstm, (h, c) = self.lstm(embeddings)
        else:
            lstm, (h, c) = self.lstm(embeddings, hidden)

        lstm = lstm.contiguous().view(-1, lstm.shape[2])
        logits = self.h2o(lstm)
        return logits, (h.detach(), c.detach())

    def sample(self, seq_len):
        """
        Sample a string of length `seq_len` from the model.
        :param seq_len [int]: String length
        :return [list]: A list of length `seq_len` that contains each token in order.
                        Tokens should be numbers from {0, 1, 2, ..., 656}.
        """
        champion_Freq = self.dataset.champion_Freq
        with torch.no_grad():
            # The starting hidden state of LSTM is None
            h_prev = None
            # Accumulate tokens into texts
            texts = []
            # Randomly draw the starting token and convert it to a torch.tensor
            x = np.random.choice(champion_Freq.shape[0], 1, p=champion_Freq)[None, :]
            x = torch.from_numpy(x).type(torch.int64).to(self.device)
            ##### Complete the code here #####
            # Append each generated token to texts, including the starting token.
            # hint: you can use self.forward
            ##################################

        return texts

    def compute_prob(self, string):
        """
        Compute the probability for each string in `strings`
        :param string [np.ndarray]: an integer array of length N.
        :return [float]: the log-likelihood
        """
        champion_Freq = self.dataset.champion_Freq
        with torch.no_grad():
            # The starting hidden state of LSTM is None
            h_prev = None
            # Convert the starting token to a torch.tensor
            x = string[None, 0, None]
            x = torch.from_numpy(x).type(torch.int64).to(self.device)
            # The log-likelihood of the first token.
            # You should accumulate log-likelihoods of all other tokens to ll as well.
            ll = np.log(champion_Freq[string[0]])
            ##### Complete the code here ######
            # Add the log-likelihood of each token into ll
            # hint: you can use self.forward
            ###################################

            return ll
