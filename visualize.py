
import torch
from tsne_torch import TorchTSNE as TSNE
from config import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def plot_embedding_space(model,data, save_path="embedding_space.png"):
    model.eval()
    # Get the embeddings
    images, texts = data
    images = images.to(device)
    token_batch = model.text.tokenize(texts, device)

    img_embs, txt_embs = model(images, **token_batch, output_embeddings=True)
    concat_embs = torch.cat((img_embs, txt_embs), dim=0)
    X_emb = TSNE(n_components=2, perplexity=30, initial_dims=embedding_dim).fit_transform(concat_embs)
    # Plot the embeddings
    color_options = list(mcolors.CSS4_COLORS.values())
    for i in range(len(X_emb) // 2):
        color = color_options[len(color_options)*i//(len(X_emb)//2)]
        plt.scatter(X_emb[i, 0], X_emb[i, 1],label=f"Image {i}", color=color, marker='x')
        plt.scatter(X_emb[i+len(X_emb) // 2, 0], X_emb[i+len(X_emb) // 2, 1], label=f"Text {i}", color=color, marker='o')
    #plt.show()
    plt.legend()
    plt.savefig(save_path)
    plt.close()

#print(mcolors.CSS4_COLORS.values())

