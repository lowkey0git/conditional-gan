# Conditional GAN for Image Generation

This repository contains a **conditional GAN (cGAN)** implementation in PyTorch that generates images conditioned on features extracted from a pre‑trained classifier (ResNet‑18). [9][10] The generator learns to produce realistic images such that the feature representation of the fake image matches that of real images from the same class, enabling class‑controlled image synthesis on your dataset. [3][9]

***

## What this project does

- **Conditional image generation**: the generator takes a latent vector $$z$$ and a class‑conditioned feature vector (from a ResNet‑18 feature extractor) and outputs a 3‑channel image. [9][10]
- **Feature‑matching loss**: the discriminator is trained with standard GAN loss while the generator is regularized with an MSE loss between feature maps of real and fake images, encouraging semantic consistency. [3][9]
- **Class‑mean conditioning**: optionally, images can be conditioned on precomputed per‑class mean features instead of per‑sample features, which can stabilize training on small datasets. [3][9]

***

## How to use

### 1. Prerequisites

You need:
- Python 3.7+ with PyTorch and torchvision installed. [7][10]
- A pre‑trained **feature extractor** (ResNet‑18 with `fc` replaced by `nn.Identity()`), saved as `feature_extractor.pt`. [4][10]

### 2. Dataset format

Organize your data as:
```text
data_dir/
  train/
    class_0/
      img001.jpg
      ...
    class_1/
      ...
```

This script expects an `ImageFolder`‑compatible structure under `data_dir/train`. [4][7]

### 3. Run training

Example command:

```bash
python train_conditional_gan.py \
  --data_dir /path/to/data_dir \
  --feature_extractor /path/to/feature_extractor.pt \
  --out_dir gan_out \
  --epochs 20 \
  --batch_size 32 \
  --img_size 128 \
  --z_dim 100 \
  --lr 2e-4 \
  --lambda_feat 10.0 \
  --use_class_mean
```

This:
- trains the conditional GAN for 20 epochs at 128×128 resolution, [4][10]
- saves checkpoints (`G_epoch{N}.pt`, `D_epoch{N}.pt`) and example images (`epoch{N}_class{K}.png`) under `gan_out`. [4][7]

### 4. Inference / sampling

To generate from a trained model, instantiate `Generator` and run `G(z, feat)` with a fixed `feat` (e.g., `class_mean[cls]` from training) and a random `z` vector. The script already dumps example images at each epoch so you can directly inspect the generated samples. [4][9]

***

## Key modules

- `Generator`: MLP‑based mapping from `z` + projected feature to a 4D image tensor, then upsampled via transposed convolutions. [4][10]
- `Discriminator`: standard CNN‑based discriminator with LeakyReLU and batch normalization. [4][10]
- `load_feature_extractor`: wraps a ResNet‑18 feature extractor whose parameters are frozen after loading. [4][10]
- `denorm_to_tanh` / `to_display` / `tensor_to_bgr`: utilities that convert between ImageNet‑normalized images and the `[-1,1]` range expected by the generator and for visualization. [3][9]

***

## License and attribution

You should add a `LICENSE` file (e.g., MIT, Apache, or another open‑source license) and credit any pre‑trained models or datasets you use.
Citations:
[1] LSGAN with GitHub and PyTorch: A Comprehensive Guide https://www.codegenes.net/blog/lsgan-github-pytorch/
[2] The Open Source Repository Guide: Best Practices for Sharing Your ... https://app.readytensor.ai/publications/best-practices-for-ai-project-code-repositories-0llldKKtn8Xb
[3] Conditional Generative Model 구현 (feat. pytorch) https://alpha.velog.io/@wodnr0710/Conditional-Generative-Model-%EA%B5%AC%ED%98%84-feat.-pytorch
[4] GitHub - frotms/GAN-Pytorch-Template: A Generative Adversarial Networks(GAN) project template to simplify building and training deep learning models using pytorch. https://github.com/frotms/GAN-Pytorch-Template
[5] readme-files – Best Practices for Writing Reproducible Code https://utrechtuniversity.github.io/workshop-computational-reproducibility/chapters/readme-files.html
[6] Pix2Pix-Conditional-GANs/train.py at main · AquibPy/Pix2Pix-Conditional-GANs https://github.com/AquibPy/Pix2Pix-Conditional-GANs/blob/main/train.py
[7] Simple GAN using PyTorch https://github.com/vamsi3/simple-GAN
[8] How to write a good README - GitHub https://github.com/banesullivan/README
[9] Implementing Conditional GANs in PyTorch for Controlled Synthesis https://www.slingacademy.com/article/implementing-conditional-gans-in-pytorch-for-controlled-synthesis/
[10] Generative Adversarial Networks (GANs) in PyTorch https://www.geeksforgeeks.org/deep-learning/generative-adversarial-networks-gans-in-pytorch/
