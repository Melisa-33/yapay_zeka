import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Hiperparametreler
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Veri Ön İşleme ve Yükleme
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# CNN Modeli
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),    # 1 giriş kanalı, 20 çıkış kanalı (28x28 -> 24x24)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24x24 -> 12x12

            nn.Conv2d(20, 50, kernel_size=5),  # 20 giriş kanalı, 50 çıkış kanalı (12x12 -> 8x8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 4 * 4, 500),  # 50x4x4 öznitelik haritasını düzleştirip 500 birime bağla
            nn.ReLU(),
            nn.Dropout(p=0.3),        
            nn.Linear(500, 10)           # 10 sınıf
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Model, Kayıp Fonksiyonu, Optimizasyon
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Eğitim ve Test Fonksiyonları
def train(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Doğruluk Hesaplama
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {correct / total:.4f}")

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Eğitim ve Test
train(model, train_loader, criterion, optimizer, num_epochs)
test(model, test_loader)

#görselleştirme
def plot_predictions(model, test_loader):
    model.eval()
    data, labels = next(iter(test_loader))  # Test verisinden bir batch al
    data, labels = data.to(device), labels.to(device)  # Veriyi GPU'ya taşı
    output = model(data)
    preds = output.argmax(dim=1, keepdim=True)

    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(data[i][0].cpu().detach().numpy(), cmap='gray')
        title_color = 'green' if preds[i].item() == labels[i].item() else 'red'
        plt.title(f"Pred: {preds[i].item()}, True: {labels[i].item()}", color=title_color)
        plt.axis('off')
    plt.show()

# Görselleştirme için çağırma
plot_predictions(model, test_loader)
