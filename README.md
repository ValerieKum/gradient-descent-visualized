# Gradient Descent: Learning Rate Race

Several learning rates race down the same loss surface, showing which converge, which crawl, and which diverge.

Part of my portfolio of small, from-scratch visualisations of computer-science ideas. Built on numpy and matplotlib, so every moving part is visible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python gradient_descent.py                  # live animated window
python gradient_descent.py --save out.gif   # export a looping GIF
python gradient_descent.py --save out.mp4   # smaller file, best for the web (needs ffmpeg)
```
