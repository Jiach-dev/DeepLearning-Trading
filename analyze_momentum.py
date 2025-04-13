import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision import transforms
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

# Load image
image_path = "AAPL.png"
image_pil = Image.open(image_path).convert("RGB")
tensor = transforms.ToTensor()(image_pil)

# Extract RGB channels
red, green, blue = tensor[0], tensor[1], tensor[2]
blue_threshold = 0.7
plot_mask = blue < blue_threshold

# Normalize
def normalize(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-6)

red_norm = normalize(red)
green_norm = normalize(green)

focused_red = red_norm * plot_mask
focused_green = green_norm * plot_mask

# Compute momentum
def compute_momentum(r, g):
    red_val = r.mean().item()
    green_val = g.mean().item()
    signal = "Bullish" if green_val > red_val else "Bearish"
    return red_val, green_val, signal

# Plot generator with colorbars
plot_images = []

def generate_plot_image(r, g, red_val, green_val, signal, stage):
    bg_color = "#ffffff"
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(bg_color)
    fig.suptitle(f"Momentum Analysis - Stage {stage}", fontsize=14, fontweight='bold', color='black')

    for ax in axs:
        ax.set_facecolor(bg_color)

    im_r = axs[0].imshow(r, cmap="Reds", interpolation="nearest")
    axs[0].set_title("Focused Red", fontsize=12, color='darkred')
    axs[0].axis("off")
    cbar_r = fig.colorbar(im_r, ax=axs[0], fraction=0.046, pad=0.04)
    cbar_r.ax.tick_params(labelsize=8)

    im_g = axs[1].imshow(g, cmap="Greens", interpolation="nearest")
    axs[1].set_title("Focused Green", fontsize=12, color='darkgreen')
    axs[1].axis("off")
    cbar_g = fig.colorbar(im_g, ax=axs[1], fraction=0.046, pad=0.04)
    cbar_g.ax.tick_params(labelsize=8)

    axs[2].bar(["Red", "Green"], [red_val, green_val], color=["#d62728", "#2ca02c"])
    axs[2].set_ylabel("Intensity", fontsize=11)
    axs[2].set_title(f"Momentum: {signal}", fontsize=13,
                     color='green' if signal == "Bullish" else 'red')
    axs[2].tick_params(axis='y', labelsize=10)
    axs[2].set_facecolor(bg_color)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plot_images.append(img)
    plt.close(fig)

# Initial stage
red_val, green_val, signal = compute_momentum(focused_red, focused_green)
generate_plot_image(focused_red, focused_green, red_val, green_val, signal, stage=0)

# Max pooling stages
pooled_red = focused_red.unsqueeze(0).unsqueeze(0)
pooled_green = focused_green.unsqueeze(0).unsqueeze(0)

for stage in range(1, 8):
    pooled_red = F.max_pool2d(pooled_red, kernel_size=2)
    pooled_green = F.max_pool2d(pooled_green, kernel_size=2)
    r = pooled_red.squeeze().detach()
    g = pooled_green.squeeze().detach()
    red_val, green_val, signal = compute_momentum(r, g)
    generate_plot_image(r, g, red_val, green_val, signal, stage=stage)

# Combine with faint dividers
divider_height = 4
divider_color = (220, 220, 220)

total_height = sum(img.height for img in plot_images) + (len(plot_images) - 1) * divider_height
max_width = max(img.width for img in plot_images)
final_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

y_offset = 0
for i, img in enumerate(plot_images):
    final_image.paste(img, (0, y_offset))
    y_offset += img.height
    if i < len(plot_images) - 1:
        draw = ImageDraw.Draw(final_image)
        draw.rectangle([0, y_offset, max_width, y_offset + divider_height], fill=divider_color)
        y_offset += divider_height

final_image.save("momentum_analysis_summary.png")
print("📸 Saved: momentum_analysis_summary.png")
