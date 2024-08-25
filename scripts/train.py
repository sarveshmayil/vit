from tqdm import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from vit import ViT

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 0.005
num_epochs = 5

# MNIST dataset
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Use only 50% of the training dataset
train_dataset.data = train_dataset.data[:len(train_dataset.data)//2]
train_dataset.targets = train_dataset.targets[:len(train_dataset.targets)//2]

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the ViT model
model = ViT(
    n_classes=10,
    image_size=(32, 32),
    hidden_dim=128,
    depth=2,
    n_heads=8,
    mlp_dim=128,
    channels=1,
    patch_size=4
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pbar = tqdm(total=num_epochs, desc="Training")

# Training loop
for epoch in range(num_epochs):
    model.train()   
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        postfix = f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
        pbar.set_postfix_str(postfix)

    pbar.update(1)
pbar.close()
print('Training finished!')

# Test loop
model.eval()
with torch.inference_mode():
    correct = 0
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        logits = model(data)
        predicted = torch.argmax(logits, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Save the model
torch.save(model.state_dict(), 'vit_mnist.pth')
print('Model saved!')
