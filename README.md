# V2IR-pix2pix

## Conda Setup

Create and activate the environment:

```bash
conda create -n llvip_pix2pix python=3.8 -y
conda activate llvip_pix2pix
```

Install PyTorch (CUDA 11.8):

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

Install remaining dependencies:

```bash
pip install flask pillow requests
```

Run the web app:

```bash
cd webapp
python app.py
```

The server will start at `http://127.0.0.1:5000`.
