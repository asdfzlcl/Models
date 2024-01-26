import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from exp2.TorchModels.GCNNet import GCNNet
from exp2.database.MyDataset import MyOwnDataset

if __name__=='__main__':
    MODEL_PATH = "model/"
    DATA_PATH = "database/kalahai.txt"
    DEVICE_ID = "cuda:1"
    torch.set_printoptions(precision=8)
    DEVICE = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH + 'D'):
        discriminator = torch.load(MODEL_PATH + 'D')
    discriminator.to(DEVICE)
    dataSet = MyOwnDataset('database', 'jiduo', discriminator, DEVICE)
    model = GCNNet().to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print(model)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        loss = torch.tensor(0.0).to(DEVICE)
        for data in dataSet:
            out = model(data)
            #print(out.shape)
            loss += F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        print(loss)