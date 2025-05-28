import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


# === DATASET ===
class MultimodalDataset(Dataset):
    def __init__(self, df, max_images=8):
        self.df = df
        self.max_images = max_images

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = {k: v.squeeze(0) for k, v in row['tokenized_claim'].items()}
        evidence = {k: v.squeeze(0) for k, v in row['tokenized_evidence'].items()}
        image_paths = row['image_path']

        images = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    # img = image_transform(img.convert("RGB"))
                    img = clip_transform(img.convert("RGB"))
                    images.append(img)
            except:
                continue
        if not images:
            images = torch.zeros((self.max_images, 3, 224, 224))
        else:
            images = torch.stack(images[:self.max_images])
            if images.shape[0] < self.max_images:
                pad = torch.zeros((self.max_images - images.shape[0], 3, 224, 224))
                images = torch.cat([images, pad], dim=0)

        label = torch.tensor(row['cleaned_truthfulness'], dtype=torch.long)
        return text, evidence, images, label

# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

clip_transform = Compose([
    Resize(224, interpolation=Image.BICUBIC),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
              std=[0.26862954, 0.26130258, 0.27577711])
])

def load_images(image_paths):
    """Load all images for a given row and return a tensor list."""
    images = []
    for path in image_paths:
        try:
            with Image.open(path) as image: # faster image loading
                image = image.convert("RGB")
                # image = image_transform(image) # appy the transformation
                image = clip_transform(image)
                images.append(image)
        except Exception as e:
            print(f"Error loading {path}: {e}")  
    return torch.stack(images) if images else torch.zeros((1, 3, 224, 224))
# if multiple images exist, stack them into a single tensor
# returns a black image when there is no image loaded
    