import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST


num_epochs = 1
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = MNIST('./data', transform=img_transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class EverFree(nn.Module):
    def __init__(self):
        super(EverFree, self).__init__()
        self.Convo = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, 5, 1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1) 
        )
        self.Fully = nn.Sequential(
            nn.Linear(4*4*50, 500),
            nn.ReLU(True),
            nn.Linear(500, 10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.Convo(x)
        x = x.view(-1, 4*4*50)
        x = self.Fully(x)
        return x


model = EverFree()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


for epoch in range(num_epochs):
    for data in dataloader:
        img, target = data
        print(target)
        img = Variable(img)
        output = model(img)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data))
