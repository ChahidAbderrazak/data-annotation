

def run_show_image_and_mask():
  image_path = '/media/abdo2020/DATA1/Datasets/images-dataset/raw-data/5.JPG'
  mask_path = 'mask.tif'

  # load the image
  image = load_image(image_path)
  mask = load_image(mask_path)
  # check if mask exists
  dir, filename, file_extension=path_split(image_path)

  # initialise viewer with coins image
  viewer = napari.view_image(image, name='imag:'+filename, rgb=False)
  # add the labels
  label_layer = viewer.add_labels(mask, name='mask')

  napari.run()



  