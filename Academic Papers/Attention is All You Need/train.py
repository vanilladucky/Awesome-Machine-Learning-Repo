from transformer import get_model
import torch
import torch.nn as nn
import torch.optim as optim
from getdata import get_data
from tqdm import tqdm

class train:
    def __init__(self):
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.n_epochs = 50
        self.device = torch.device('cuda')

    def forward(self):
        self.model = get_model()
        self.model.train()
        self.optim = optim.Adam(self.model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)
        self.model = self.model.to(self.device)
        self.criterion.to(self.device)
        for epoch in tqdm(range(self.n_epochs)):
            self.data = get_data(bs = 64)
            train_loss = [0]
            count = [0]
            self.optim.zero_grad()
            try:
                while True:
                    eng, fre = next(self.data)
                    eng = eng.to(self.device)
                    fre = fre.to(self.device)
                    output = self.model(eng, fre[:, :-1])
                    loss = self.criterion(output.contiguous().view(-1, 24053), fre[:, 1:].contiguous().view(-1))
                    loss.backward()
                    self.optim.step()
                    if count % 10 == 0:
                        print(f"==== Loss: {loss.item()} ====")
                    train_loss[0] += loss.item()
                    count[0]+=1
            except:
                print(f"==== Average loss: {train_loss[0]/count[0]} ====")
                continue

t = train()
t.forward()