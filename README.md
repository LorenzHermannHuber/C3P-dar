# C3P-dar: Conditional Poisson Point Processes for Image-to-Radar Generation

This repository contains the code accompanying the article **C3P-dar:
Conditional Poisson Point Processes for Image-to-Radar Generation**.

> **Provisional release:** This is a temporary code release accompanying the
> article. A revised and expanded release will be available soon.

## Code contributors

This code was developed jointly by
**Lorenz Hermann Huber** ([lorenz.h.huber@campus.tu-berlin.de](mailto:lorenz.h.huber@campus.tu-berlin.de))
and **Jonas Neuhöfer** ([jonas.neuhoefer@aumovio.com](mailto:jonas.neuhoefer@aumovio.com)).

## Setup

To load the source code of [RadarGen](https://github.com/tomerborreda/RadarGen), on which we show the benefit of our C3P-dar loss, use `git submodule update --init third_party/RadarGen`. Alternatively to initialise all submodules including [UFM](https://github.com/UniFlowMatch/UFM/), [UniCeption](https://github.com/castacks/UniCeption/) and [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) use

```bash
git submodule update --init --recursive
```

Create a virtual environment with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch and torchvision from your preferred supplier, e.g. a CUDA build from the [PyTorch installation selector](https://pytorch.org/get-started/locally/). Our experiments used torch 2.5.1 and torchvision 0.20.1 with CUDA 12.4:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```
