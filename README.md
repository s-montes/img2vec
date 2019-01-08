# Get vector representations of images

## Uses:
- Clustering of images
- Classification
- Check similarity

## Contains:
### img_to_vec.py

(Based on: https://github.com/christiansafka/img2vec)

Vector representations obtained using transfer learning. The implemention is done using PyTorch.

#### Architectures (resulting vector size):
- alexnet (4096)
- resnet-18 (512)
- resnet-152 (2048)
- vgg19 (4096)

### skimage_embeddings

Vector representation obtained using the Histogram of Oriented Gradient (HOG) feature descriptor. 