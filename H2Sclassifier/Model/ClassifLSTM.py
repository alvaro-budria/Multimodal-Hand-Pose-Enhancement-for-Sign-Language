import torch.nn as nn

class ClassifLSTM(nn.Module):

    def __init__(self, hidden_size, num_layers, seq_len, batch_size, num_rotations, NUM_CLASSES, bidirectional=True, dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.NUM_CLASSES = NUM_CLASSES
        self.bidirectional = bidirectional
        self.dropout = dropout

        # Define the LSTM layer
        self.lstm = nn.LSTM(num_rotations, hidden_size, num_layers, bidirectional=self.bidirectional, dropout=self.dropout, batch_first=True)

        # Define a Linear Layer to obtain the depth coordinate
        self.Linear = nn.Linear(hidden_size*(1+self.bidirectional), NUM_CLASSES)

    def forward(self, seq, state=None):
        h, state = self.lstm(seq, state) # h.shape = [batch_size, seq_len, hidden_size]
        h = h.contiguous().view(-1, self.hidden_size*(1+self.bidirectional)) # h.shape = [batch_size*seq_len, hidden_size]
        y = self.Linear(h) # y.shape = [batch_size*seq_len, NUM_CLASSES]
        y = y.contiguous().view(self.batch_size, self.seq_len, self.NUM_CLASSES) #y.shape = [batch_size, seq_len, NUM_CLASSES]
        return y, state
