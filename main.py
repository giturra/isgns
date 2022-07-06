import torch
from data import TweetStream
from preprocessing import Collate
from model import SkipGram

from torch.utils.data import DataLoader

def accuracy(pred):
    pred = (torch.sigmoid(pred) > 0.5).int()
    target = torch.LongTensor([1, 0, 0]).to(device)
    correct = ((pred == target).sum(dim=1) == 3)
    return correct.sum() / len(correct)

ts = TweetStream('C:/Users/gabri/Desktop/biwv/biwv/proccess_tweets.txt')

col = Collate(neg_samples_sum=2)

dataloader = DataLoader(ts, batch_size=64, collate_fn=col)

device = torch.device('cpu')

VOCAB_SIZE = col.vocab.max_size
EMBEDDING_DIM = 100

model = SkipGram(VOCAB_SIZE, EMBEDDING_DIM)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
criterion = torch.nn.BCEWithLogitsLoss()

N_EPOCH = 1
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, 
#     max_lr=110.,  # from LR range test plot
#     epochs=N_EPOCH, 
#     steps_per_epoch=len(dataloader)
# )

# for X, y in dataloader:
#     # X = batch[0].to(device)
#     # y = batch[1].to(device)
#     print(X)
#     print(y)
#     break
losses, accs = [], []
for i in range(1, N_EPOCH+1):
    loss_epoch = 0.
    acc_epoch = 0.
    for batch, target in dataloader: #, position=0, leave=True, desc=f"Epoch {i:03}"):
        
        x = batch.to(device)
        target = target.to(device)
        model.zero_grad()
        pred = model(x)
        loss = criterion(pred.float(), target.float())
        loss.backward()
        optimizer.step()
        #scheduler.step()
        
        loss_epoch += loss.item()
        acc_epoch += accuracy(pred).item()
    
    losses.append(loss_epoch)
    accs.append(acc_epoch)
    
    if i % 1 == 0:
        print(f"epoch: {i:03}, loss: {loss_epoch: .3f}, acc: {acc_epoch: .4f}")