import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

#Dataset Class:Loads + labels
class SimpleImageDataset(Dataset):
    def __init__(self, in_domain_path):
        #Transform
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        self.images = []
        self.labels = []

        #Read 
        classes = [d for d in sorted(os.listdir(in_domain_path))
                   if os.path.isdir(os.path.join(in_domain_path, d))]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        #Load 
        for class_name in classes:
            class_path = os.path.join(in_domain_path, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


# CNN
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        #extraction layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        #connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x  # logits


# Training 
def learn(path_to_in_domain, path_to_out_domain):
    """
    Train a model using only in-domain data.
    The out-domain path is accepted to match the assignment interface,
    but intentionally not used in training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load 
    dataset = SimpleImageDataset(path_to_in_domain)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    #int model
    model = SimpleCNN(num_classes=len(dataset.class_to_idx))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    #10 epochs
    for epoch in range(10):
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return model


# Accuracy Evaluation
def compute_accuracy(path_to_eval_folder, model):
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    class EvalDataset(Dataset):
        def __init__(self, eval_path):
             #same preprocessing as training
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
            ])

            self.images = []
            self.labels = []

            classes = [d for d in sorted(os.listdir(eval_path))
                       if os.path.isdir(os.path.join(eval_path, d))]
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

            for class_name in classes:
                class_path = os.path.join(eval_path, class_name)
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(class_path, img_name)
                        self.images.append(img_path)
                        self.labels.append(class_to_idx[class_name])

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            #apply&return
            image = Image.open(self.images[idx]).convert("RGB")
            image = self.transform(image)
            label = self.labels[idx]
            return image, label

    eval_dataset = EvalDataset(path_to_eval_folder)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    correct = 0
    total = 0
    #no gradients
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if total == 0:
        return 0.0 #avoid div by 0

    return correct / total
