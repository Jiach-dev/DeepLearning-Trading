import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os

# Step 1: Load image and convert to RGB tensor
image_path = os.path.join("charts", "AAPL.png")
image_pil = Image.open(image_path).convert("RGB")
tensor = transforms.ToTensor()(image_pil)

# Step 2: Extract RGB channels
red = tensor[0, :, :]
green = tensor[1, :, :]
blue = tensor[2, :, :]

# Step 3: Create mask for candlestick area using blue channel
blue_threshold = 0.7
mask = blue < blue_threshold

# Step 4: Focused red and green regions
focused_red = red * mask
focused_green = green * mask

# Step 5: Calculate summed intensities
def compute_momentum(r, g):
    red_sum = r.sum().item()
    green_sum = g.sum().item()
    signal = "Bullish" if green_sum > red_sum else "Bearish"
    return red_sum, green_sum, signal

# Step 6: Visualization
def plot_focus(r, g, red_sum, green_sum, signal, stage):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), facecolor="#f9f9f9")
    axs[0].imshow(r, cmap="Reds")
    axs[0].set_title(f"🟥 Focused Red - Stage {stage}", color="red", fontsize=12)
    axs[0].axis("off")

    axs[1].imshow(g, cmap="Greens")
    axs[1].set_title(f"🟩 Focused Green - Stage {stage}", color="green", fontsize=12)
    axs[1].axis("off")

    axs[2].bar(["Red", "Green"], [red_sum, green_sum], color=["red", "green"])
    axs[2].set_title(f"Momentum @ Stage {stage} → {signal}", color="green" if signal == "Bullish" else "red", fontsize=12)
    axs[2].set_ylabel("Intensity Sum")

    plt.tight_layout()
    plt.show()

# Stage 0
red_sum, green_sum, signal = compute_momentum(focused_red, focused_green)
plot_focus(focused_red, focused_green, red_sum, green_sum, signal, stage=0)

# MaxPooling
pooled_red = focused_red.unsqueeze(0).unsqueeze(0)
pooled_green = focused_green.unsqueeze(0).unsqueeze(0)

for stage in range(1, 8):
    pooled_red = F.max_pool2d(pooled_red, kernel_size=2)
    pooled_green = F.max_pool2d(pooled_green, kernel_size=2)
    r = pooled_red.squeeze().detach()
    g = pooled_green.squeeze().detach()
    red_sum, green_sum, signal = compute_momentum(r, g)
    plot_focus(r, g, red_sum, green_sum, signal, stage=stage)
