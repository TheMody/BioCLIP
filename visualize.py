
import torch
from tsne_torch import TorchTSNE as TSNE
from config import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def plot_embedding_space(model,test_loader_imgs,test_loader_ecg, save_path="embedding_space.png"):
    model.eval()
    concat_non_text = []
    concat_text = []
    img_iter = iter(test_loader_imgs)
    ecg_iter = iter(test_loader_ecg)
    for i in range(gradient_accumulation_steps):
        images, texts = next(img_iter)
        images = images.to(device)
        token_batch = model.text.tokenize(texts, device)
        data1 = {"images": images, "text": token_batch}

        ecg, texts = next(ecg_iter)
        ecg = ecg.to(device)
        token_batch = model.text.tokenize(texts, device)
        data2 = {"ecg": ecg, "text": token_batch}

        data = []
        data.append(data1)
        data.append(data2)
        # Get the embeddings

        for subset in data:
            non_text_embs, txt_embs = model(**subset, output_embeddings=True)
            concat_non_text.append(non_text_embs)
            concat_text.append(txt_embs)
       # concat_embs = torch.cat((img_embs, txt_embs), dim=0)
      #  concat_embs_all.append(concat_embs)
    concat_embs = torch.cat(concat_non_text, dim=0)
    concat_embs_text = torch.cat(concat_text, dim=0)
    concat_embs = torch.cat((concat_embs, concat_embs_text), dim=0)

    X_emb = TSNE(n_components=2, perplexity=30, initial_dims=PROJ_DIM).fit_transform(concat_embs)
    # Plot the embeddings
    color_options = list(mcolors.CSS4_COLORS.values())
    for i in range(len(X_emb) // 2):
        color = color_options[len(color_options)*i//(len(X_emb)//2)]
        plt.scatter(X_emb[i, 0], X_emb[i, 1],label=f"Non text modality {i}", color=color, marker='x')
        plt.scatter(X_emb[i+len(X_emb) // 2, 0], X_emb[i+len(X_emb) // 2, 1], label=f"Text {i}", color=color, marker='o')
    #plt.show()
 #   plt.legend()
    plt.savefig(save_path)
    plt.close()

#print(mcolors.CSS4_COLORS.values())

