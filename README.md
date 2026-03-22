# V2IR-pix2pix

V2IR-pix2pix is a web application that translates visible-light images and videos into their infrared equivalents using a pix2pix GAN model trained on the LLVIP dataset. The generator is a U-Net 256 architecture trained for 200 epochs, producing 256x256 infrared outputs from RGB inputs. Before translation, input images are upscaled using a Stable Diffusion upscaling model to enhance detail and improve the quality of the generated infrared output.

The application supports uploading local image or video files, as well as loading images directly from a URL. For images, it runs the IR translation and optionally performs object detection on the output using a Roboflow model. For videos, it processes the footage frame by frame, reassembles the translated frames into a downloadable video, and applies detection with rate limiting to avoid hitting API limits. The input and output are displayed side by side for easy comparison.

## Conda Setup

Create and activate the environment:

```bash
conda create -n v2ir python=3.10 -y
conda activate v2ir
```

Install PyTorch (CUDA 12.9):

```bash
conda install pytorch torchvision pytorch-cuda=12.9 -c pytorch -c nvidia -y
```

Install remaining dependencies:

```bash
pip install flask pillow requests
```

Run the web app:

```bash
python app.py
```

The server will start at `http://127.0.0.1:5000`.
