
# Face Recognition (Classification) & Face Verification (Cosine Similarity)

This project is to explore how to extract important features from face images that can be used to
effectively verify identities. The network's performance is evaluated with two tasks:
- face classification: identify the name of the person in the image
- face verification: tell if two images are of same person or not

## Key Ideas Explored
- CNN Architectures: ConvNeXt, ResNet, ...
- Data Augmentation: 
    - torchvision.transforms.v2（flip, small rotation, color jitter）
    - MixUp
    - CutMix
- Face Recognition Loss Functions:
    - Classification-oriented: Softmax CrossEntropy
    - Feature-based: Triplet, N-Pair, ...
    - Margin-based Softmax: ArcFace, CosFace, SphereFace, AM-Softmax, ...
    - Unified: Circle, SupCom, ...
    - Train with 2x loss functions: add, alternatively, ...

---

## Data Layout

The notebook assumes the following structure:

### Classification
```
cls_data_dir/
  train/
    images/
    labels.txt
  dev/
    images/
    labels.txt
  test/
    images/
    labels.txt
```

Each `labels.txt` line:
```
relative_image_path label_id
```

### Verification
- `ver_data_dir/` contains the verification images
- `val_pairs.txt` format:
```
img_path1 img_path2 match(0/1)
```
- `test_pairs.txt` format:
```
img_path1 img_path2
```

Metrics: **ACC / EER / AUC / TPR@FPR**

---

## Configuration

Key tunables (tune it by yourself):
- model architecture: ConvNeXtV2-style network (depthwise conv + MLP blocks), global average pooling
- `batch_size = 256`
- `lr = 0.004`
- `epochs = 50`
- `image_size = 112`
- optimizer: `AdamW(lr=0.004, weight_decay=0.05)`
- scheduler: `ReduceLROnPlateau(factor=0.5, patience=5, mode="max", min_lr=1e-4)` monitoring `valid_cls_acc`

---

## Acknowledgement
The original starter codes and writeup remain the intellectual property of the CMU Introduction to Deep Learning course professor and teaching assistants and are subject to the course’s copyright and licensing terms. Please do not use or distribute any part of this repository in ways that violate course policies or academic integrity guidelines.
