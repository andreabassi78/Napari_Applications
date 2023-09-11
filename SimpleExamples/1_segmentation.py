
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, remove_small_objects
import matplotlib.pyplot as plt

coins = data.coins()
thresh = threshold_otsu(coins)
closing_size = 4
bw = closing(coins > thresh, square(closing_size))
cleared = remove_small_objects(clear_border(bw), 20)
label_image = label(cleared)


plt.imshow(coins)

plt.imshow(label_image)

plt.show()