import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from lib.utils import yaml2config
from networks import get_model
from lib.alphabet import strLabelConverter

def preprocess_image(image_path, img_height=32):
    # Load image in grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize keeping aspect ratio, setting height to img_height
    w, h = img.size
    new_w = int(w * (img_height / h))
    img = img.resize((new_w, img_height), Image.BILINEAR)
    
    # Standard FW-GAN normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img_tensor = transform(img).unsqueeze(0) # [1, 1, H, W]
    return img_tensor

def main():
    parser = argparse.ArgumentParser(description="Generate handwriting with custom style and text.")
    parser.add_argument("--config", type=str, default="./configs/fw_gan_iam.yml", help="Path to config file")
    parser.add_argument("--style_image", type=str, required=True, help="Path to the reference handwriting image")
    parser.add_argument("--text", type=str, required=True, help="Text to generate")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the generated image")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    args = parser.parse_args()

    # Load config
    cfg = yaml2config(args.config)
    cfg.device = args.device
    
    # Initialize Model (reusing AdversarialModel for simplicity in loading)
    logdir = "temp_infer"
    model_class = get_model(cfg.model)
    model = model_class(cfg, logdir)
    
    # Load weights
    if os.path.exists(cfg.ckpt):
        print(f"Loading checkpoint from {cfg.ckpt}")
        model.load(cfg.ckpt, cfg.device)
    else:
        print(f"Error: Checkpoint not found at {cfg.ckpt}")
        print("Please download FW-GAN.pth and place it in data/weights/")
        return

    model.set_mode('eval')
    device = torch.device(args.device)

    # Preprocess style image
    style_img = preprocess_image(args.style_image, cfg.img_height).to(device)
    style_len = torch.IntTensor([style_img.shape[-1]]).to(device)

    # Encode style
    with torch.no_grad():
        # Encode style using StyleEncoder (E) and SharedBackbone (S)
        enc_styles = model.models.E(style_img, style_len, model.models.S)
        
        # Add noise as done in the paper/code
        noise_dim = cfg.GenModel.style_dim - cfg.EncModel.style_dim
        noises = torch.randn((1, noise_dim)).to(device)
        enc_z = torch.cat([noises, enc_styles], dim=-1)

        # Generate each word one by one and stitch them together
        words = args.text.split(' ')
        fake_imgs_list = []
        alphabet = model.label_converter.alphabet

        for i, word in enumerate(words):
            if not word: continue
            
            # Filter text to only include characters in the alphabet
            clean_word = ''.join(c for c in word if c in alphabet)
            if not clean_word:
                print(f"Warning: No valid characters in word: '{word}'")
                continue
            
            # Use dummy second item to force batch mode
            word_labels, word_label_lens = model.label_converter.encode([clean_word, alphabet[1]])
            word_labels = word_labels[0:1].to(device)
            word_label_lens = word_label_lens[0:1].to(device)
            
            # Generate word image
            word_img = model.models.G(enc_z, word_labels.long(), word_label_lens.int())
            
            # Crop to actual width
            word_width = word_label_lens[0].item() * cfg.char_width
            word_img = word_img[:, :, :, :word_width]
            fake_imgs_list.append(word_img)
            
            # Add space between words (except for the last word)
            if i < len(words) - 1:
                # 16 pixels is a standard space width in the original code's paragraph generation
                space_width = 16 
                space = torch.ones(1, 1, cfg.img_height, space_width).to(device)
                fake_imgs_list.append(space)

        if not fake_imgs_list:
            print("Error: No words could be generated.")
            return

        # Stitch words together
        full_img = torch.cat(fake_imgs_list, dim=-1)

        # Post-process: Convert from [-1, 1] to [0, 255]
        img_out = full_img[0, 0].cpu().numpy()
        img_out = 255 * ((img_out + 1) / 2)
        
        # Darken the ink: 
        # Since background is white (255) and ink is dark, 
        # we can apply a power transformation to push gray values towards black.
        # Gamma < 1 makes it lighter, Gamma > 1 makes it darker (for black ink).
        # Actually, for black ink on white background, applying a power to normalized pixels [0, 1]
        # where ink is near 0 and background is near 1:
        # img_norm^2 will keep 1 near 1 and make 0.5 become 0.25 (darker).
        img_out = np.clip(img_out, 0, 255).astype(np.float32) / 255.0
        img_out = np.power(img_out, 1.5) # Apply Gamma to darken
        img_out = (img_out * 255.0).astype(np.uint8)
        
        # Optional: Linear contrast stretch
        # img_out = cv2.normalize(img_out, None, 0, 255, cv2.NORM_MINMAX)
        
        cv2.imwrite(args.output, img_out)
        print(f"Generated stitched image saved to {args.output}")

if __name__ == "__main__":
    main()
