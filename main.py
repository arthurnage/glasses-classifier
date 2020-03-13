import cv2
import mobilenetv3
import sys
import time
import torch
import torch.nn as nn

from glob import glob
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from facenet_pytorch import MTCNN


class DataFromPaths(Dataset):
    def __init__(self, paths, transform=None):
        super(DataFromPaths).__init__()
        self.paths = paths
        self.transform = transform
    
    def __getitem__(self, i):
        p = self.paths[i]
        img = cv2.imread(p)
        img = img[:, :, ::-1]
        if self.transform is not None:
            img = self.transform(img)
            
        return img, p
        
    def __len__(self):
        return len(self.paths)


def main():
    images_folder = sys.argv[1]
    input_images = glob(f"{images_folder}/*")
    input_images = [i for i in input_images if i.split('.')[-1] in ['jpg', 'png']]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(128, device=device)

    for p in input_images:
        image_id = p.split('/')[-1]
        image = cv2.imread(p)
        mtcnn(image, save_path=f"{images_folder}_faces/{image_id}")

    input_faces = glob(f"{images_folder}_faces/*")
    del mtcnn

    tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
        ])
    
    data = DataFromPaths(input_faces, transform=tr)
    loader = DataLoader(data, batch_size=64)

    net = mobilenetv3.mobilenetv3_small()

    mod = nn.Sequential(
        net.features[:10],
        net.conv,
        net.avgpool,
        nn.Flatten(),
        nn.Linear(in_features=576, out_features=2)
    )

    mod.load_state_dict(torch.load('best.pth'))
    mod.to(device)
    mod.eval()

    for batch, paths in loader:
        batch = batch.to(device)
        outputs = mod(batch)
        _, preds = torch.max(outputs, 1)
        
        preds = preds.cpu().numpy()
        
        for pred, path in zip(preds, paths):
            if pred == 0:
                print(path.replace('_faces', ''))

if __name__ == "__main__":
    main()
