import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import clip
import ssl

# Uncomment if you experience SSL context issue
# ssl._create_default_https_context = ssl._create_stdlib_context

def extract_clip_features(image_dir, output_file, batch_size=64, device=None):
    """
    Extract image features using CLIP model and save them to a file.
    
    Args:
        image_dir (str): Directory containing images.
        output_file (str): File path to save the extracted features.
        batch_size (int): Number of images to process in a batch.
        device (torch.device): Device to run the model on.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = clip.load("ViT-B/32", device=device, download_root='./CLIP')
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    image_filenames = [
        os.path.join(root, file)
        for root, _, files in os.walk(image_dir)
        for file in files if file.endswith('.png')
    ]

    all_features = []
    for i in tqdm(range(0, len(image_filenames), batch_size), desc="Extracting Features"):
        batch_images = [
            preprocess(Image.open(image_filenames[j]).convert('RGB'))
            for j in range(i, min(i + batch_size, len(image_filenames)))
        ]
        batch_images = torch.stack(batch_images).to(device)

        with torch.no_grad():
            batch_features = model.encode_image(batch_images)
            all_features.append(batch_features.cpu().numpy())

    np.savez(output_file, clip_features=np.concatenate(all_features, axis=0))
    print(f"Features saved to {output_file}")

if __name__ == "__main__":
    image_dir = f'<Images Directory>'
    output_file = f'<Output Directory>'
    extract_clip_features(image_dir, output_file)