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
python main.py infer <path/to/content-image> <path/to/style-image> --model_path <path/to/model-weights> --alpha <alpha> --samples <samples>
```

## Observations

On inference, the images are upsampled by default to a 2048x2048 shape using bilinear interpolation before being passed to the model.
This I found to provide the best resulting output images when downsampled back to the original size of the content images.
That said, 1024x1024 will also work to produce minimal artifacting on medium-sized images but going higher will usually produce better results.

|Content|Style|Result|
|:---:|:---:|:---:|
|<img src="./sample_images/content/000000003136.jpg" width="500px"> | <img src="./sample_images/style/24.jpg" width="500px">| <img src="./sample_images/result/3136-24.png" width="500px">|
|<img src="./sample_images/content/000000000155.jpg" width="500px"> | <img src="./sample_images/style/107.jpg" width="500px">| <img src="./sample_images/result/155-107.png" width="500px">|
|<img src="./sample_images/content/000000002122.jpg" width="500px"> | <img src="./sample_images/style/171.jpg" width="500px">| <img src="./sample_images/result/2122-171.png" width="500px">|
