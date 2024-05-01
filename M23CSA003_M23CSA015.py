# %%

# # IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# # TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# # THEN FEEL FREE TO DELETE THIS CELL.
# # NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# # ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# # NOTEBOOK.

# import os
# import sys
# from tempfile import NamedTemporaryFile
# from urllib.request import urlopen
# from urllib.parse import unquote, urlparse
# from urllib.error import HTTPError
# from zipfile import ZipFile
# import tarfile
# import shutil

# CHUNK_SIZE = 40960
# DATA_SOURCE_MAPPING = 'dl-a5-data:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F4891196%2F8244595%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240427%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240427T120724Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D6a3816b3d48e7d5edb3cf8a77807c18c838bb7afa3db50b70f97ea92f0fd9b523ae8df6de239e46d9087a1af4bcc53fbd76bcafe13e817d8b145be08050d95071928add94265be1c21b2b419367d1d7fa6e3fed379af9eb8a49f189180654bb815a8b20d2b07c96e56ef02ec418e1b3bb40a24dd8fcfa76fcd5208946a96baeb349182b1ba4e4e48a768547d71725753a52e35c5211a17d1b185c5a3f46cf5d04a46a1e61b5a43866520cf4a5aab5db0cfc671dfc5220f798e2d2544f458e48a585c9a4790e32c75172d7f742307ce34ed0242faca29f08ca11333a5006488fad0afa720127eb0ca6f90bfff36f908ae0bcb409c83a91ccac8e1490d1f83bb11'

# KAGGLE_INPUT_PATH='/kaggle/input'
# KAGGLE_WORKING_PATH='/kaggle/working'
# KAGGLE_SYMLINK='kaggle'

# !umount /kaggle/input/ 2> /dev/null
# shutil.rmtree('/kaggle/input', ignore_errors=True)
# os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
# os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

# try:
#   os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
# except FileExistsError:
#   pass
# try:
#   os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
# except FileExistsError:
#   pass

# for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
#     directory, download_url_encoded = data_source_mapping.split(':')
#     download_url = unquote(download_url_encoded)
#     filename = urlparse(download_url).path
#     destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
#     try:
#         with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
#             total_length = fileres.headers['content-length']
#             print(f'Downloading {directory}, {total_length} bytes compressed')
#             dl = 0
#             data = fileres.read(CHUNK_SIZE)
#             while len(data) > 0:
#                 dl += len(data)
#                 tfile.write(data)
#                 done = int(50 * dl / int(total_length))
#                 sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
#                 sys.stdout.flush()
#                 data = fileres.read(CHUNK_SIZE)
#             if filename.endswith('.zip'):
#               with ZipFile(tfile) as zfile:
#                 zfile.extractall(destination_path)
#             else:
#               with tarfile.open(tfile.name) as tarfile:
#                 tarfile.extractall(destination_path)
#             print(f'\nDownloaded and uncompressed: {directory}')
#     except HTTPError as e:
#         print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
#         continue
#     except OSError as e:
#         print(f'Failed to load {download_url} to path {destination_path}')
#         continue

# print('Data source import complete.')


# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
#!pip3 install -U -r requirements.txt

# %%
# from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


# from six.moves import xrange

# import umap
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Load Data

# %%
import os
from glob import glob1

# from torchvision.io import read_image
from PIL.Image import open as open_image


class CustomImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.img_dir = (
            glob1(root, "*.jpg") + glob1(root, "*.png") + glob1(root, "*.jpeg")
        )
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.img_dir[idx])
        image = open_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image


# %% [markdown]
# ### Finding Mean and Variance of data

# %%
# training_data = CustomImageDataset(root=r"/kaggle/input/dl-a5-data/Train_data-001/Train_data", transform=transforms.Compose([ transforms.CenterCrop(128),transforms.ToTensor(), transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))]))
# train_loader = DataLoader(training_data, batch_size=256, shuffle=False)
# # placeholders
# psum = torch.tensor([0.0, 0.0, 0.0])
# psum_sq = torch.tensor([0.0, 0.0, 0.0])

# # loop through images
# imgs = []
# for inputs in tqdm(train_loader):
#     psum += inputs.sum(axis=[0, 2, 3])
#     psum_sq += (inputs**2).sum(axis=[0, 2, 3])
#     imgs.append(inputs)
# data_isic = torch.cat(imgs, dim=0)
# data_variance = torch.var(data_isic)


# %%
# ####### FINAL CALCULATIONS

# # pixel count
# count = len(training_data) * 128 * 128

# # mean and std
# total_mean = psum / count
# total_var = (psum_sq / count) - (total_mean**2)
# total_std = torch.sqrt(total_var)

# # output
# print("mean: " + str(total_mean))
# print("std:  " + str(total_std))
# print('var' , data_variance)

# %%


# %%
mean = (0.6567, 0.3680, 0.3743)
std = (0.1701, 0.1709, 0.1831)
data_variance = 0.0487
# root = "/kaggle/input/dl-a5-data/"
root = r"D:/programming/Assignment_5"
# root = "/kaggle/input/dl-assignment5"
training_data = CustomImageDataset(
    root=root + "/Train_data",
    transform=transforms.Compose(
        [
            transforms.RandomHorizontalFlip(0.4),
            transforms.RandomRotation(10),
            # transforms.RandomCrop(128, padding=4, padding_mode='reflect'),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    ),
)

validation_data = CustomImageDataset(
    root=root + "/Test/Test_data",
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    ),
)

# %% [markdown]
# ## Vector Quantizer Layer


# %%
class VectorQuantizerEMA(nn.Module):
    def __init__(
        self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5
    ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


# %% [markdown]
# ## Encoder & Decoder Architecture
#
# The encoder and decoder architecture is based on a ResNet and is implemented below:


# %%
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [
                Residual(in_channels, num_hiddens, num_residual_hiddens)
                for _ in range(self._num_residual_layers)
            ]
        )

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


# %%
class Encoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


# %%
class Decoder(nn.Module):
    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
        )

        self._conv_trans_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self._conv_trans_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=3,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


# %% [markdown]
# # Initialization
#
# We use the hyperparameters from the author's code:

# %%
batch_size = 256
num_training_updates = 15000

num_hiddens = 64  # 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 128
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

# %%
training_loader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
)

# %%
validation_loader = DataLoader(
    validation_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=0
)


# %%
class Model(nn.Module):
    def __init__(
        self,
        num_hiddens,
        num_residual_layers,
        num_residual_hiddens,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        decay=0,
    ):
        super(Model, self).__init__()

        self._encoder = Encoder(
            3, num_hiddens, num_residual_layers, num_residual_hiddens
        )
        self._pre_vq_conv = nn.Conv2d(
            in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1
        )

        self._vq_vae = VectorQuantizerEMA(
            num_embeddings, embedding_dim, commitment_cost, decay
        )

        self._decoder = Decoder(
            embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens
        )

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity


# %%
model = Model(
    num_hiddens,
    num_residual_layers,
    num_residual_hiddens,
    num_embeddings,
    embedding_dim,
    commitment_cost,
    decay,
)
ngpu = torch.cuda.device_count()
if ngpu > 1:
    model = nn.DataParallel(model)
model = model.to(device)


# %%
def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation="nearest")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def denorm(img_tensors, mean, std):
    stats_image = (mean, std)
    return img_tensors * stats_image[1][0] + stats_image[0][0]


# %% [markdown]
# # Training

# %%
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# %%
model.train()
train_res_recon_error = []
train_res_perplexity = []
epochs = 50
for i in range(epochs):
    for data in tqdm(training_loader):
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        # break

        # if (i+1) % 100 == 0:

    print(f"epoch: {i+1}/{epochs} ")
    print("%d iterations" % ((i + 1) * len(training_loader)))
    print("recon_error: %.3f" % np.mean(train_res_recon_error[-100:]))
    print("perplexity: %.3f" % np.mean(train_res_perplexity[-100:]))
    show(make_grid(denorm(data[:16].cpu().data, mean, std)))
    plt.show()
    show(make_grid(denorm(data_recon[:16].cpu().data, mean, std)))
    plt.show()

    state_dict = {
        "model": model.state_dict(),
        "epochs": i,
        "recon_error": np.mean(train_res_recon_error[-100:]),
        "perplexity": np.mean(train_res_perplexity[-100:]),
    }
    torch.save(state_dict, "/content/drive/MyDrive/DL_A5/model_vqvae.pth")
    print("model saved")
    print()
    # break


# %%


# %% [markdown]
# ## Plot Loss

# %%
train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
train_res_perplexity_smooth = savgol_filter(train_res_perplexity, 201, 7)

# %%
f = plt.figure(figsize=(16, 8))
ax = f.add_subplot(1, 2, 1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale("log")
ax.set_title("Smoothed NMSE.")
ax.set_xlabel("iteration")

ax = f.add_subplot(1, 2, 2)
ax.plot(train_res_perplexity_smooth)
ax.set_title("Smoothed Average codebook usage (perplexity).")
ax.set_xlabel("iteration")

# %% [markdown]
# ## View Reconstructions

# %%
model.eval()
model.load_state_dict(torch.load("./model_vqvae.pth")["model"])
(valid_originals) = next(iter(validation_loader))
valid_originals = valid_originals.to(device)

vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
_, valid_quantize, _, encodings = model._vq_vae(vq_output_eval)
valid_reconstructions = model._decoder(valid_quantize)

# %%
# Reconstructions
show(make_grid(denorm(valid_reconstructions.cpu().data, mean, std)))

# %%
# Originals
show(make_grid(denorm(valid_originals.cpu().data, mean, std)))

# %% [markdown]
# ## View Embedding

# %%
# torch.save(model.state_dict(), 'model_vqvae_e50.pth')

# %%
# !pip install umap-learn
import umap

# %%


# %%
proj = umap.UMAP(n_neighbors=4, min_dist=0.1, metric="cosine").fit_transform(
    model._vq_vae._embedding.weight.data.cpu()
)

# %%
plt.scatter(proj[:, 0], proj[:, 1], alpha=0.3)

# %% [markdown]
# # Auto Regressive Generation

# %%
model.load_state_dict(torch.load("./model_vqvae.pth")["model"])

# %%
# model(input)
model.eval()

# %%
torch.onnx.export(
    model, torch.randn(1, 3, 128, 128).to(device), "model.onnx", verbose=True
)

# %%
embedding_dim = 128
num_embeddings = 512

# %% [markdown]
# ## Random Latent Codebook Permutations

# %%
N = 32

input_shape = (N, 32, 32, 128)

# %%

z = torch.randint(0, num_embeddings, (N * 32 * 32,)).unsqueeze(1).to(device)
encodings = torch.zeros(z.shape[0], num_embeddings, device=device)
encodings.scatter_(1, z, 1)
# Quantize and unflatten
print(z.shape)
quantized = (
    torch.matmul(encodings, model._vq_vae._embedding.weight)
    .view(input_shape)
    .permute(0, 3, 1, 2)
)

x_recon = model._decoder(quantized)
show(make_grid(denorm(x_recon.cpu().data, mean, std)))

# %%
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 128  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 600000
eval_interval = 500
learning_rate = [4e-4, 4e-4, 0.00005]  # change lr every 200000 steps
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 128  # embedding dimension
n_head = 6
n_layer = 6
dropout = 0.2
vocab_size = 512 + 2  # number of embeddings in the VQ-VAE
torch.manual_seed(1337)

# %%


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTRegressiveModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # This token embedding is already fixed
        # self.token_embedding_table  = vq_vae_embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# %%
batch_size = 1
training_loader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
)
validation_loader = DataLoader(
    validation_data, batch_size=1, shuffle=True, pin_memory=True, num_workers=0
)

# %%
valid_originals = next(iter(training_loader))
# data = data.to(device)
input_shape = (batch_size, 32, 32, 128)
valid_originals = valid_originals.to(device)

vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
_, valid_quantize, _, encodings = model._vq_vae(vq_output_eval)
valid_reconstructions = model._decoder(valid_quantize)

# %%
show(make_grid(denorm(valid_reconstructions.cpu().data, mean, std)))

# %%
start = torch.tensor([512], dtype=torch.long).to(device)
end = torch.tensor([513], dtype=torch.long).to(device)

# %%
import wandb

wandb.login(key="5d1c2f2e3eed3439166d8e749b48bad14e6854f8")

# %%

modelgpt = GPTRegressiveModel()
m = modelgpt.to(device)
loaded_state_dict = torch.load(
    "/kaggle/input/fork-of-notebook37b2edd6e5/modelgpt_resume.pth"
)
m.load_state_dict(loaded_state_dict["model"])

run = wandb.init(
    project="VQVAE",
    entity="khadgaa",
    name="modelgpt_3_05",
    notes="modelgpt resume training from 4e6 to 6e6 iters. lr=5e-5",
)

wandb.watch(modelgpt)

# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(modelgpt.parameters(), lr=learning_rate)
optimizer.load_state_dict(loaded_state_dict["optimizer"])


@torch.no_grad()
def estimate_loss(training_loader, validation_loader):
    out = {}
    modelgpt.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                valid_originals = next(iter(training_loader))
            else:
                valid_originals = next(iter(validation_loader))
            # data = data.to(device)
            valid_originals = valid_originals.to(device)
            vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
            _, valid_quantize, _, encodings = model._vq_vae(vq_output_eval)
            valid_reconstructions = model._decoder(valid_quantize)
            encodings = torch.concat(
                (start, encodings.argmax(dim=-1).flatten(), end), dim=0
            )
            X, Y = get_batch(encodings)
            logits, loss = modelgpt(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    modelgpt.train()
    return out


# # data loading
def get_batch(data):
    # generate a small batch of data of inputs x and targets y
    # data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


input_shape = (batch_size, 32, 32, 128)


for iters in tqdm(range(max_iters)):

    # every once in a while evaluate the loss on train and val sets
    if iters % eval_interval == 0 or iters == max_iters - 1:
        losses = estimate_loss(training_loader, validation_loader)
        print(
            f"step {iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        wandb.log(losses)
        # sample a batch of data

        valid_originals = next(iter(training_loader))
        # data = data.to(device)

        valid_originals = valid_originals.to(device)

        vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
        _, valid_quantize, _, encodings = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(valid_quantize)
        encodings = torch.concat(
            (start, encodings.argmax(dim=-1).flatten(), end), dim=0
        )
        xb, yb = get_batch(encodings)

        # evaluate the loss
        logits, loss = modelgpt(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        state_dict = {
            "model": modelgpt.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iters:": iters,
        }
        torch.save(state_dict, "/kaggle/working/modelgpt_resume_3.pth")


# %% [markdown]
# ## Auto Regressive model Inference

# %%
modelgpt = GPTRegressiveModel()
m = modelgpt.to(device)
loaded_state_dict = torch.load("./modelgpt_resume_3.pth")
m.load_state_dict(loaded_state_dict["model"])

# %%
N = 32
with torch.no_grad():
    generated_encodes = []
    context = torch.tensor([[512]] * N, dtype=torch.long, device=device)
    generated_encode = m.generate(context, max_new_tokens=1025)

generated_encode = generated_encode[:, 1:-1]
mode, _ = torch.mode(generated_encode, dim=1)
generated_encode = torch.where(
    generated_encode == 512, mode.unsqueeze(1), generated_encode
)
generated_encode = torch.where(
    generated_encode == 513, mode.unsqueeze(1), generated_encode
)

# %%
generated_encodes_ = generated_encode.reshape(-1, 1)

# %%
input_shape = (N, 32, 32, 128)
encodings = torch.zeros(generated_encodes_.shape[0], num_embeddings, device=device)
encodings.scatter_(1, generated_encodes_, 1)
quantized = (
    torch.matmul(encodings, model._vq_vae._embedding.weight)
    .view(input_shape)
    .permute(0, 3, 1, 2)
)

x_recon = model._decoder(quantized)
show(make_grid(denorm(x_recon.cpu().data, mean, std)))

# %%
