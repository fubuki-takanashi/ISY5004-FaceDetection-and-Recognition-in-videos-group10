import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

class WiderFaceDataset(Dataset):
    def __init__(self, images_dir, annotations_file, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_paths = []
        self.bboxes = []

        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            filename = lines[i].strip()
            image_path = os.path.join(images_dir, filename)
            num_faces = int(lines[i + 1].strip())
            boxes = []
            for j in range(i + 2, i + 2 + num_faces):
                box = list(map(int, lines[j].strip().split()[:4]))
                boxes.append(box)
            i = i + 2 + num_faces
            
            self.image_paths.append(image_path)
            self.bboxes.append(boxes)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = np.array(self.bboxes[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, boxes

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, targets

def train_model(images_dir, annotations_file):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),  
        transforms.ToTensor()
    ])

    dataset = WiderFaceDataset(images_dir, annotations_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    optimizer = torch.optim.Adam(mtcnn.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    num_epochs = 10
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for images, targets in dataloader:
            images = images.to(device)
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for img, target in zip(images, targets):
                boxes, _ = mtcnn.detect(img)
                if boxes is not None:
                    target = torch.tensor(target, dtype=torch.float).to(device)
                    if target.size(0) == boxes.size(0):
                        loss = criterion(boxes, target)
                        batch_loss += loss.item()
                        loss.backward()
            
            optimizer.step()
            epoch_loss += batch_loss / len(images)
        
        losses.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    plt.plot(range(1, num_epochs + 1), losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    images_dir = r'D:\Document\5004\video_store\Wider_face\WIDER_train\images'
    annotations_file = r'D:\Document\5004\video_store\Wider_face\wider_face_split\wider_face_train_bbx_gt.txt'
    train_model(images_dir, annotations_file)
