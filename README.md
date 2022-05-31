# AdaIN on TensorFlow

A barebones TensorFlow implementation of a style transfer neural network using adaptive instance normalization (AdaIN)  
Uses a pretrained VGG19 for the encoder, closely following _Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization_ by Xun Huang, Serge Belongie

## Usage

Train with COCO dataset (available [here](https://cocodataset.org/#download)) as content images and WikiArt dataset (available [here](https://www.kaggle.com/competitions/painter-by-numbers/data)) as style images.  
Run training with command:

```bash
python main.py train <path/to/content-dir> <path/to/style-dir>
```

Current script will save only the model weights by default as a Keras HDF5 file (.h5), rather than a TensorFlow SavedModel.

---
For sampling on inference, run:

```bash
python main.py infer <path/to/content-image> <path/to/style-image> --model_path <path/to/model-weights> --alpha <alpha> --mixing <mixing-ratio>
```

## Observations

On inference, the images are upsampled by default to a 4096x4096 shape using bilinear interpolation before being passed to the model.
This I found to provide the best resulting output images when downsampled back to the original size of the content images.
That said, 2048x2048 will also work to produce minimal artifacting on medium-sized images and 1024x1024 upsampling will also work on small images.

Results pending...

_Trained on TF v2.6.0_
