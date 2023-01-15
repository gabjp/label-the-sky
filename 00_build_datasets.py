from label_the_sky.preprocessing._datagen import get_dataset
from pandas import read_csv
import numpy as np
import os
from skimage import io
import sys


if len(sys.argv) != 4:
    print('usage: python {} <csv_file> <n_channels> <target>'.format(
        sys.argv[0]))
    exit(1)

csv_file = sys.argv[1]
n_channels = int(sys.argv[2])
target = sys.argv[3]

if n_channels not in [3, 12]:
    raise ValueError('n_channels must be: 3, 12')

if target not in ['class', 'magnitudes']:
    raise ValueError('target must be: class, magnitudes')

base_dir = os.path.join(os.environ['HOME'], 'raw_fits')
data_dir = 'crops_rgb32' if n_channels==3 else 'all_objects'
data_dir = os.path.join(base_dir, data_dir)
output_dir = os.path.join(os.environ['HOME'], 'data')

dataset_name = csv_file.split('/')[-1][:-4]
dtype = 'uint8' if n_channels==3 else 'float32'
read_fn = io.imread if n_channels==3 else np.load
ext = '.png' if n_channels==3 else '.npy'

df = read_csv(csv_file)

for split in ['train', 'val', 'test']:
    df_tmp = df[(df.split==split)]
    ids, y, labels = get_dataset(df_tmp, target=target, n_bands=n_channels)
    np.save(
        os.path.join(
            output_dir,
            f'{dataset_name}_{n_channels}_y_{split}.npy'),
        y)
    print(f'saved {dataset_name}_{n_channels}_y_{split}.npy', y.shape)
    im_paths = [os.path.join(data_dir, i + ext) for i in ids]
    X = np.zeros((len(ids),) + (32, 32, n_channels), dtype=dtype)
    for i, path in enumerate(im_paths):
        im_tmp =  read_fn(path) #(12,32,32)
        if n_channels == 12: # Lazy fix to shape problem
            X[i, :] = np.transpose(im_tmp, (1,2,0))        #(32,32,12)
        else:
            print(im_tmp)
            X[i, :] = im_tmp
            

    np.save(
        os.path.join(
            output_dir,
            f'{dataset_name}_{n_channels}_X_{split}.npy'),
        X)
    print(f'saved {dataset_name}_{n_channels}_X_{split}.npy', X.shape)
