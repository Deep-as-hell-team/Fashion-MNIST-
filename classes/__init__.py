import numpy as np

np.random.seed(1337)  # for reproducibility
IMG_ROWS, IMG_COLS = 28, 28
np.set_printoptions(precision=2)
classes = np.array(['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
