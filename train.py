import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.transforms import ToTensor, Normalize, Compose, Resize
from torchsummary import summary
from tqdm import tqdm
from preprocessing import Crowd11
from crowd_behaviour.action_model import DeeplyLearnedAttributes

model = DeeplyLearnedAttributes()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

crowd_dataset = Crowd11(root_dir='data/crowd11/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f'Training Loss: {epoch_loss:.4f}')