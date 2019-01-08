import numpy as np
import sys
import os
import pickle

from skimage import data
from skimage import io

from PIL import Image

from skimage.color import rgb2gray
from skimage.transform import resize

from skimage.filters.rank import mean_bilateral
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.feature import hog


def hog_embeddings_to_pickle(input_path, output_path = None, mode = 'normal'):
    ''' Create vector embeddings using entropy + HOG
    input_path : Input folder
    output_path : Output folder for pickles (default: same as input)
    mode : 'normal' (regular image), 'entropy' (entropy image)
    '''
    output_path = output_path or input_path
    
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    
    sample_size = len(files)
    sample_indices = list(range(sample_size))
    
    file = files[0]
    filename = os.fsdecode(file)
        
    # Read image
    img = load_square_image(os.path.join(input_path, filename))
    
    # Resize
    img = resize(img, (224, 224), mode='constant', anti_aliasing=True)
        
    if mode == 'entropy':
        # Convert to grayscale
        img = rgb2gray(img)
        img = np.array(img*255, dtype = np.uint16)
    
        # Entropy
        img = entropy(img, disk(2))
        
        multichannel = False
    else:
        multichannel = True
        
    # HOG embedding
    vec = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), block_norm ="L2-Hys", multichannel = multichannel)
    
    print("Encoding size: {0}".format(vec.shape[0]))
    
    vec_mat = np.zeros((sample_size, vec.shape[0]))
    
    for index, i in enumerate(sample_indices):
                       
        if (i % 100 == 0): print("Iteration: {0}".format(i))
        file = files[i]
        filename = os.fsdecode(file)
        
        # Read image
        img = load_square_image(os.path.join(input_path, filename))
        
        # Resize
        img = resize(img, (224, 224), mode='constant', anti_aliasing=True)
        
        
        if mode == 'entropy':
            # Convert to grayscale
            img = rgb2gray(img)
            img = np.array(img*255, dtype = np.uint16)
    
            # Entropy
            img = entropy(img, disk(2))
        
            multichannel = False
        else:
            multichannel = True
        
        # HOG embedding
        vec = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), block_norm ="L2-Hys", multichannel = multichannel)
        
        vec_mat[index,:] = vec
        
    output_name = output_path + '/' + mode + '-hog' + '_img_names.pkl'
    print('Processed filenames: '+ output_name)
    pickle.dump(files, open(output_name, 'wb'))
        
    output_name = output_path + '/' +  mode + '-hog' + '_img_vecs.pkl'
    print('Embedded vectors: '+ output_name)
    pickle.dump(vec_mat, open(output_name, 'wb'))
    
def load_square_image(ImageFilePath):
    
    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if(width != height):
        bigside = width if width > height else height

        background = Image.new('RGBA', (bigside, bigside), (0, 0, 0, 0))
        offset = (int(round(((bigside - width) / 2), 0)), int(round(((bigside - height) / 2),0)))

        background.paste(image, offset)
        return np.array(background)
        #print("Image has been resized !")

    else:
        return np.array(image)
        #print("Image is already a square, it has not been resized !")