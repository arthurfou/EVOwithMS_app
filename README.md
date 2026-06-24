# EVOwithMS — Event Visual Odometry with Motion Segmentation

**Arthur Fournier** — IPAL (CNRS/NUS), Singapore

Gradio demo app for the pipeline: **Motion Segmentation → Visual Odometry (DEVO) → Results**.
Supports both live inference and replay of saved results.

## Overview

This app visualizes the integration of a motion segmentation module with DEVO (Deep Event Visual Odometry).
The goal is to detect dynamic objects in event camera streams and filter them out before pose estimation.

### Tabs

| Tab | Status | Description |
|-----|--------|-------------|
| **Filtrage EVIMO** | Ready | Render a `.bag` sequence as APS + event frames side by side |
| **EV-IMO Filtered — Events** | Ready | Compare original vs. filtered events (side-by-side video + stats) |
| **Motion Segmentation** | Coming soon | Visualize dynamic mask/score output from `MS_Model` |
| **DEVO Odometry** | Coming soon | Visualize DEVO trajectory with/without motion segmentation |

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate it

```bash
conda activate EVOwithMS
```

### 3. Launch the app

```bash
python app.py
```

The app starts a local Gradio server and prints a public `share` link in the terminal.

## Repository structure

```
EVOwithMS/
├── app.py                  # Entry point (Gradio Blocks)
├── environment.yml         # Conda environment
├── app/
│   ├── bag_reader.py       # ROS bag loading + video generation
│   ├── evs_reader.py       # Filtered event loading + comparison frames
│   └── tabs/
│       ├── evimo_viewer.py # Tab: Filtrage EVIMO
│       └── evs_viewer.py   # Tab: EV-IMO Filtered — Events
├── DEVO/                   # DEVO fork reference
└── datasets/
    ├── EV-IMO/             # Raw EV-IMO sequences (.bag)
    ├── EV-IMO_filtered/    # Filtered sequences (evs.npy + tss_imgs_us.txt)
    └── filter0/
```

## Context

This repo is part of a research internship at IPAL on motion segmentation for event-based visual odometry.
See the parent [`../CLAUDE.md`](../CLAUDE.md) for the full scientific context.
