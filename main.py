"""
Simple CLIP‑like training skeleton using:
  - Vision encoder: DINOv2 ViT‑B/14 (timm)
  - Text encoder: DistilBERT (HuggingFace Transformers)

This script defines the encoders, a CLIP wrapper with symmetric InfoNCE loss, and a minimal
training loop that trains on CIFAR‑100 images with class names as pseudo‑captions. Replace
the dataset section with your own image–text pairs (e.g. COCO, Flickr30k) to use it in practice.
"""



from config import *
import argparse
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
import time
from torchvision.datasets import CIFAR100
from torchvision import transforms
from model import CLIP
import wandb
from tqdm import tqdm
from scheduler import CosineWarmupScheduler
from visualize import plot_embedding_space
from data import make_cifar_collate_fn#, make_ecg_collate_fn
from data_ecg import PTBXLWaveformDataset
import math
import torch.nn.functional as F

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = "cuda" if torch.cuda.is_available() else "cpu"



def clip_loss(logits: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE loss (cross‑entropy in both directions). Normalized to be invariant to the batch size""" 
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i = F.cross_entropy(logits, targets) 
    loss_t = F.cross_entropy(logits.t(), targets) 
    return ((loss_i + loss_t) / 2)/ math.log(logits.size(0))

@torch.no_grad()
def evaluate(model: CLIP, dataloader_img: DataLoader, dataloader_ecg:DataLoader):
    """Compute loss and top‑1 retrieval accuracy in both directions."""
    model.eval()
    total_loss = 0.0
    correct_i2t = 0  # image‑>text retrieval accuracy
    correct_t2i = 0  # text‑>image retrieval accuracy
    n_samples = 0

    for images, texts in tqdm(dataloader_img, desc="Evaluating images"):
        images = images.to(device)
        texts = model.text.tokenize(texts, device)

        logits = model(images = images, text = texts)
        loss = clip_loss(logits)
        total_loss += loss.item()

        targets = torch.arange(logits.size(0), device=device)
        pred_i = logits.argmax(dim=1)
        pred_t = logits.argmax(dim=0)
        correct_i2t += (pred_i == targets).sum().item()
        correct_t2i += (pred_t == targets).sum().item()
        n_samples += logits.size(0)

    for timeseries, texts in tqdm(dataloader_ecg, desc="Evaluating ecg"):
        timeseries = timeseries.to(device)
        texts = model.text.tokenize(texts, device)

        logits = model(ecg = timeseries, text = texts)
        loss = clip_loss(logits)
        total_loss += loss.item()

        targets = torch.arange(logits.size(0), device=device)
        pred_i = logits.argmax(dim=1)
        pred_t = logits.argmax(dim=0)
        correct_i2t += (pred_i == targets).sum().item()
        correct_t2i += (pred_t == targets).sum().item()
        n_samples += logits.size(0)

    mean_loss = total_loss / len(dataloader_img)
    acc_i2t = correct_i2t / n_samples
    acc_t2i = correct_t2i / n_samples
    return mean_loss, acc_i2t, acc_t2i

def main():
    parser = argparse.ArgumentParser(description="Train a minimal CLIP using DINOv2 + DistilBERT on CIFAR‑100.")
    parser.add_argument("--output", type=Path, default=Path("models/bio_clip.pth"))
    args = parser.parse_args()


    model = CLIP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMG_SIZE),  # ensures exact H=W=518
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    trainset_imgs = CIFAR100(root="data", train=True, download=True, transform=transform)

    class_names = trainset_imgs.classes  # fine‑label names (CIFAR‑100 has 100)

   # print(class_names)
    testset_imgs = CIFAR100(root="data", train=False, download=True, transform=transform)

    train_loader_imgs = DataLoader(trainset_imgs, batch_size=batch_size*gradient_accumulation_steps, shuffle=True,
                              collate_fn=make_cifar_collate_fn(class_names), num_workers=4, pin_memory=True)
    test_loader_imgs = DataLoader(testset_imgs, batch_size=batch_size, shuffle=False,
                             collate_fn=make_cifar_collate_fn(class_names), num_workers=4, pin_memory=True)
    

    trainset_ecg = PTBXLWaveformDataset(root="data/ptbxl", split="train")

    testset_ecg = PTBXLWaveformDataset(root="data/ptbxl", split="test")

    train_loader_ecg = DataLoader(trainset_ecg, batch_size=batch_size*gradient_accumulation_steps, shuffle=True, num_workers=4, pin_memory=True)
    test_loader_ecg = DataLoader(testset_ecg, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    steps_per_epoch = min(len(train_loader_imgs), len(train_loader_ecg))
    scheduler = CosineWarmupScheduler(optimizer, warmup=100, max_iters=epochs*steps_per_epoch, min_factor=0.1)

    wandb.init(project="bioclip", config=args)
    min_loss = 1e10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        with tqdm(total=steps_per_epoch, desc="Training,epoch " + str(epoch) ) as pbar:
            img_loader = iter(train_loader_imgs)
            ecg_loader = iter(train_loader_ecg)
           # for images, texts in train_loader_imgs:
            for i in range(steps_per_epoch):
                start_time = time.time()
                images, texts_img = next(img_loader)
                images = images.to(device)
                timeseries, texts_ecg = next(ecg_loader)
                timeseries = timeseries.to(device)
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                start = 0
                log_dict = {}
                for a in range(gradient_accumulation_steps*NUM_MODALITIES):
                    end = start + batch_size
                    sub_images =  sub_texts =  sub_ecgs = None
                    if a % NUM_MODALITIES == 0:
                        sub_images = images[start:end]
                        sub_texts = texts_img[start:end]
                    else:
                        sub_ecgs = timeseries[start:end]
                        sub_texts = texts_ecg[start:end]
                    token_batch = model.text.tokenize(sub_texts, device)
                    logits = model(images = sub_images, text = token_batch, ecg = sub_ecgs )
                    loss = clip_loss(logits) / (gradient_accumulation_steps)  # scale 
                    loss.backward()
                    if a % NUM_MODALITIES == 0:
                        log_dict["img_loss"] = loss.item() * gradient_accumulation_steps
                    else:
                        log_dict["ecg_loss"] = loss.item() * gradient_accumulation_steps
                    batch_loss += loss.item()  /NUM_MODALITIES
                    start = (a //NUM_MODALITIES) * batch_size
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                total_loss += batch_loss

                log_dict["loss"] = batch_loss
                log_dict["lr"] = scheduler.get_last_lr()[0]
                log_dict["time"] = time.time() - start_time
                wandb.log(log_dict)
                pbar.update(1)
                pbar.set_postfix(loss=batch_loss)
        total_loss /= steps_per_epoch #step#
        print(f"Epoch {epoch + 1}/{epochs} \t average loss over epoch: {total_loss:.4f}")
        plot_embedding_space(model, test_loader_imgs,test_loader_ecg)
        mean_loss, acc_i2t, acc_t2i = evaluate(model, test_loader_imgs, test_loader_ecg)
        print(f"Test loss: {mean_loss:.4f} \t i2t acc: {acc_i2t:.4f} \t t2i acc: {acc_t2i:.4f}")
        wandbimg = wandb.Image("embedding_space.png")
        wandb.log({"test_loss": mean_loss, "acc_i2t": acc_i2t, "acc_t2i": acc_t2i, "embedding_space": wandbimg})
        if mean_loss < min_loss:
            min_loss = mean_loss
            print(f"Saving model with loss {min_loss:.4f}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)


if __name__ == "__main__":
    main()
