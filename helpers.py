import os
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F
from torchviz import make_dot
import pandas as pd
import ast
import hiddenlayer as hl
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

sns.set_theme(context="talk", style="whitegrid")

CMAPS = {
    "confusion": "crest",    # perceptually uniform
    "diverging": "icefire",  # for error or deviation plots
    "bars": "viridis",       # for accuracy and metrics
}
palette = sns.color_palette("viridis", n_colors=10).as_hex()
FIGSIZE = {
    "full": (6.4, 3.6),
    "half": (3.1, 1.8),
    "tall": (5.0, 4.0),
    "wide": (7.0, 3.5),
}

plt.rcParams.update({
    # --- Figure layout ---
    "figure.figsize": FIGSIZE["full"],      # 16:9 aspect ratio (full width)
    "figure.dpi": 200,                 # crisp display, lighter files than 300
    "savefig.dpi": 300,                # for exported plots (publication quality)

    # --- Font sizes ---
    "font.size": 10,                   # match Beamer base font
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # --- Fonts ---
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Fira Sans"],
    "text.latex.preamble": r"""
        \usepackage{FiraSans}
        \usepackage{sfmath}
        \renewcommand*\familydefault{\sfdefault}
    """,

    # --- Axes and grid ---
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.3,

    # --- Save options ---
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

plotly_template = dict(
    layout=dict(
        font=dict(family="Fira Sans", size=10, color="black"),
        title=dict(font=dict(size=11)),
        xaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        yaxis=dict(title_font=dict(size=10), tickfont=dict(size=9)),
        legend=dict(font=dict(size=9)),
        width=1280,   # full slide width (~16:9)
        height=720,   # full slide height
        coloraxis=dict(colorbar=dict(title="Importance")),  # optional default
    )
)


# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

# Folder to save images
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

def savefig(name, ext="pdf", tight=True, transparent=True, dpi=300):
    """
    Save the current matplotlib figure in a clean, consistent format.

    Parameters
    ----------
    name : str
        Base filename (no extension).
    ext : str, optional
        File extension, e.g. 'pdf', 'png', or 'svg'. Default is 'pdf'.
    tight : bool, optional
        Apply tight_layout() before saving. Default is True.
    transparent : bool, optional
        Use transparent background (useful for LaTeX Beamer overlays). Default is False.
    dpi : int, optional
        Dots per inch for raster formats (e.g., PNG). Default is 300.
    """

    # Ensure output directory exists
    os.makedirs(IMG_DIR, exist_ok=True)
    
    if tight:
        plt.tight_layout()
    path = os.path.join(IMG_DIR, f"{name}.{ext}")
    plt.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)

    plt.close()
    print(f"💾 Saved: {path}")


# -------------------------------------------------------------------------
# Load trained model
# -------------------------------------------------------------------------
model = torch.load("09 best model", weights_only=False)
model.to(device)
model.eval()


# -------------------------------------------------------------------------
# 1️⃣ Predictions Grid
# -------------------------------------------------------------------------
def show_predictions_grid(model, loader, rows=5, cols=8):
    """Show a compact grid of predictions vs true labels."""
    model.eval()
    imgs, labels = next(iter(loader))
    imgs, labels = imgs.to(device), labels.to(device)

    with torch.no_grad():
        preds = model(imgs).argmax(1)

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=FIGSIZE["full"],  # base 16:9 size
        constrained_layout=True,  # use full space efficiently
    )

    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Compute number of images to show
    num_imgs = min(rows * cols, len(imgs))

    for i in range(num_imgs):
        ax = axes[i]
        img = denormalize(imgs[i])
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        # Hide all spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Disable grid
        ax.grid(False)

        true, pred = CLASSES[labels[i]], CLASSES[preds[i]]
        color = "green" if pred == true else "red"
        ax.set_title(pred, color=color, fontsize=8.5, pad=2, fontweight="semibold")

    # Hide unused axes
    for ax in axes[num_imgs:]:
        ax.axis("off")

    # Adjust spacing
    plt.subplots_adjust(wspace=0.2, hspace=0.25)

    savefig("predictions_grid")

# -------------------------------------------------------------------------
# Accuacries
# -------------------------------------------------------------------------
df = pd.read_csv("training_results.csv", converters={
        "train_accs": ast.literal_eval,
        "val_accs": ast.literal_eval,
        "lrs": ast.literal_eval
    })
df_expanded = pd.concat([
        pd.DataFrame({
            "epoch": range(1, len(r.train_accs)+1),
            "train_acc": r.train_accs,
            "val_acc": r.val_accs,
            "trial": r.trial,
            "lr": r.lrs, 
        })
        for _, r in df.iterrows()
    ], ignore_index=True)
best_trial = df.loc[df['val_acc'].idxmax(), 'trial']

def plot_accuracies(df):
    """
    Plots training and validation accuracy curves for all trials, 
    highlighting the best trial.
    
    Args:
        df (pd.DataFrame): DataFrame containing training results, with columns:
            ['trial', 'train_accs', 'val_accs', 'lrs', ...]
        save_path (str): File path to save the figure.
    """

    # All trials in faint lines
    sfig, ax = plt.subplots(1, 2, sharey=True)
    sns.lineplot(data=df_expanded, x="epoch", y="train_acc", hue="trial", alpha=0.3, legend=False, ax=ax[0])
    sns.lineplot(data=df_expanded, x="epoch", y="val_acc", hue="trial", alpha=0.3, legend=False, ax=ax[1])
    # highlight best trial
    sns.lineplot(data=df_expanded[df_expanded.trial==best_trial], x="epoch", y="train_acc", color="red", lw=2, ax=ax[0], label=f"Best trial")
    sns.lineplot(data=df_expanded[df_expanded.trial==best_trial], x="epoch", y="val_acc", color="red", lw=2, ax=ax[1], label=f"Best trial")

    ax[0].set_title("Training Accuracy")
    ax[1].set_title("Validation Accuracy")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Accuracy")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy")
    savefig("all_trials_accuracy")