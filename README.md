# Visual Momentum Analyzer

This project explores momentum analysis in candlestick charts using computer vision techniques powered by PyTorch and Matplotlib.

Red and green pixel intensities are extracted from chart images to simulate bearish vs bullish momentum. The signal is visualized at multiple scales using CNN-style max pooling.

## Features

- RGB-based intensity extraction from candlestick charts
- Momentum visualization at progressive resolutions
- Simple yet intuitive output for potential use in HFT strategies

## How to Run

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Place a candlestick chart image inside the `charts/` folder (e.g. `AAPL.png`).

3. Run the script:
    ```bash
    python analyze_momentum.py
    ```

## Example Output

- Focused red and green regions
- Momentum bar charts at each stage
