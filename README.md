# Satellite Image Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mirjan-Ali-Sha/Satellite-Image-Classification/blob/main/Satellite_Image_Classification.ipynb)

## Table of Contents

1. [Dataset](README.md#dataset)
2. [Model Building](README.md#model-building)
3. [Model Results](README.md#model-results)
4. [Conclutions](README.md#conclutions)


## Dataset

I take this dataset from Kaggle Dataset ([Semantic segmentation of aerial imagery](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)).

### About Dataset *(taken from data source)*:
Context:<br>
Humans in the Loop is publishing an open access dataset annotated for a joint project with the Mohammed Bin Rashid Space Center in Dubai, the UAE.

Content: <br>
The dataset consists of aerial imagery of Dubai obtained by MBRSC satellites and annotated with pixel-wise semantic segmentation in 6 classes. The total volume of the dataset is 72 images grouped into 6 larger tiles. The classes are:

1. Building: #3C1098
2. Land (unpaved area): #8429F6
3. Road: #6EC1E4
4. Vegetation: #FEDD3A
5. Water: #E2A929
6. Unlabeled: #9B9B9B

Acknowledgements<br>
The images were segmented by the trainees of the Roia Foundation in Syria.<br>

Tiles are saved like below;
```
'Tile 1'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
'Tile 2'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
'Tile 3'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
'Tile 4'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
'Tile 5'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
'Tile 6'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
'Tile 7'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
'Tile 8'
  |_ images (include 9 RGB images)
  |_ masks (include 9 RGB Mask files)
```

### Preparing Dataset
At first I upload all datasets to my Google Drive to smooth file management irrespective of run-time environments (CPU/GPU/Reconnect). The dataset/images are not in consistant size (hieght * width) and also few image size are very big as well. So, I decide to create Patches of 256 * 256 to reduce Memory consumption during training and for that I use `patchify` and I use `sklearn.preprocessing.MinMaxScaler` to Re-Scale only image values (for better accuracy) to 0-1 floating point (it can create more varities). You can see those codes inside the Pyhon Notebook but here I attached those codes as well;
```
from google.colab import drive
drive.mount('/content/drive')
```
```
minmaxscaler = MinMaxScaler()
img_patch_size = 256
dataset_root = '/content/drive/MyDrive/'
dataset_folder = "Dubai_Arial_Datasets"
```
Patches and Dataset Creation:
```
image_dataset = []
mask_dataset = []

for image_type in ['images' , 'masks']:
  if image_type == 'images':
    image_extension = 'jpg'
  elif image_type == 'masks':
     image_extension = 'png'
  for tile_id in range(1,8):
    for image_id in range(1,20): # we can use minimum values as well instead of 20 in range.
      image = cv2.imread(f'{dataset_root}/{dataset_folder}/Tile {tile_id}/{image_type}/image_part_00{image_id}.{image_extension}', 1)
      if image is not None: # as we given the range to avoid 
        if image_type == 'masks':
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # because imread read images in BGR format (IMREAD_COLOR_BGR)
        size_x = (image.shape[1]//img_patch_size)*img_patch_size
        size_y = (image.shape[0]//img_patch_size)*img_patch_size
        image = Image.fromarray(image)
        image = image.crop((0,0, size_x, size_y))
        image = np.array(image)
        patched_images = patchify(image, (img_patch_size, img_patch_size, 3), step=img_patch_size)
        for i in range(patched_images.shape[0]):
          for j in range(patched_images.shape[1]):
            if image_type == 'images':
              individual_patched_image = patched_images[i,j,:,:]
              # Re-scale Image values 0-1.
              individual_patched_image = minmaxscaler.fit_transform(individual_patched_image.reshape(-1, individual_patched_image.shape[-1])).reshape(individual_patched_image.shape)
              individual_patched_image = individual_patched_image[0]
              image_dataset.append(individual_patched_image)
            elif image_type == 'masks':
              individual_patched_mask = patched_images[i,j,:,:]
              individual_patched_mask = individual_patched_mask[0]
              mask_dataset.append(individual_patched_mask)
```

