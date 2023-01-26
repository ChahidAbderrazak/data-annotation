"""
Perform a segmentation and annotate the results with
bounding boxes and text
"""
import os
import sys
import numpy as np
from glob import glob
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
import napari

def progresss_bar(nb_iter):
    '''
    create a progress bare display of maxim iteration loop = <nb_iter> 
    '''
    import progressbar
    bar = progressbar.ProgressBar(maxval=nb_iter, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    return bar
    
def path_split(path):
  dir = os.path.dirname(path)
  filename, file_extension = os.path.splitext( os.path.basename(path))
  
  return dir, filename, file_extension

def load_image(path):
    import numpy as np
    import cv2
    # print(' The selected image is :', path)
    filename, file_extension = os.path.splitext(path)
    try:
        img = cv2.imread(path)
    except:
        msg = '\n Error: the image path ' + path + 'cannot be loaded!!!!'
        raise Exception(msg)
    return img

def create_new_directory(DIR):
  if not os.path.exists(DIR):
    os.makedirs(DIR)

def save_array_to_gray(arr, filename, invert=False, scale=255):
    '''
    save a 2D array to png grayscale image 
    arr : input 2D numpy array
    filename : input image file name
    scale : save image in a different scale. Default [0, 255]
    '''
    import cv2
    if len(arr.shape) == 2:
        create_new_directory(os.path.dirname(filename) )
        image_int = np.array(255*arr, np.uint8)
        if invert:
          image_int = ~image_int
        cv2.imwrite(filename, image_int)
    else:
        msg = f'\n\n--> Error: the input array dimension [{arr.shape}] is not a 2D image!!!'
        raise Exception(msg)

def get_file_tag(path):
    TAG = os.path.basename(os.path.dirname(path)) + '--' + os.path.basename(path)
    return TAG

def get_folder_tag_from_folder(path):
    if path =='' :
        return path
    else:
        TAG = os.path.basename(os.path.dirname(path)) + '--' + os.path.basename(path)
    return TAG.replace('.', '')

def get_folder_tag_from_file(path):
    if path =='' :
        return path
    else:
        TAG = os.path.basename(os.path.dirname(os.path.dirname(path))) + '--' + os.path.basename(os.path.dirname(path))
    return TAG.replace('.', '')

def segment(image):
    """Segment an image using an intensity threshold determined via
    Otsu's method.
    Parameters
    ----------
    image : np.ndarray
        The image to be segmented
    Returns
    -------
    label_image : np.ndarray
        The resulting image where each detected object labeled with a unique integer.
    """
    # apply threshold
    thresh = threshold_otsu(image)
    bw = closing(image > thresh, square(4))

    # remove artifacts connected to image border
    cleared = remove_small_objects(clear_border(bw), 20)

    # label image regions
    label_image = label(cleared)

    return label_image

def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents
    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]
    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect

def circularity(perimeter, area):
    """Calculate the circularity of the region
    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region
    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity

def get_image_mask_paths(dst, filename, img_ext='.jpg', mask_ext='.png'):
    '''
    get the image, mask pair  paths 
    dst: folder where the files are saved 
    filename: pair (image, mask) filename
    img_ext : image extention. default='.jpg'
    mask_ext : image extention. default='.png'
    '''
    img_path = os.path.join(dst, 'images', filename + img_ext)
    mask_path = os.path.join(dst, 'masks', filename + mask_ext)
    return img_path, mask_path

def preapre_annotation_folder(image_path_list, dst, invert=False, img_ext='.jpg', mask_ext='.png'):
    '''
    preapre annotation folder
    '''
    print(f'\n ###############################################################')
    print(f' ###                 preparing annotation folder             ###')
    print(f' ##############################################################')
    print(f' \n - datset size : {len(image_path_list)} images')
    cnt_img =0
    cnt_msk =0
    bar = progresss_bar(len(image_path_list))
    for k, image_path in enumerate(image_path_list):
        bar.update(k)
        # check if mask exists
        dir, filename, file_extension=path_split(image_path)
        # get the image, mask pair  paths 
        img_path, mask_path = get_image_mask_paths(dst, filename, img_ext=img_ext, mask_ext=mask_ext)
        
        # save images
        if not os.path.exists(img_path):
            # load the image
            image = load_image(image_path)
            save_array_to_gray(image, img_path, invert=invert)
            cnt_img+=1
        # save masks
        if not os.path.exists(mask_path):
            # show anntation 
            label_image = np.empty_like(image)
            label_image[ image < -np.inf] = 1#mask[ mask > 0]
            # set the class color
            label_image = label_image.astype(int)*(0) # label 0 is black in napari
            save_array_to_gray(label_image, mask_path)
            cnt_msk+=1
    print(f' \n - updates: {cnt_img} images, {cnt_msk} masks')
    return 0

def create_bounding_boxes(image, label_image):
  print(f'\n ########################################')
  print(f' ###        create bounding boxes       ###')
  print(f' ########################################')
  # create the properties dictionary
  properties = regionprops_table(
      label_image, properties=('label', 'bbox', 'perimeter', 'area')
  )
  properties['circularity'] = circularity(
      properties['perimeter'], properties['area']
  )

  # create the bounding box rectangles
  bbox_rects = make_bbox([properties[f'bbox-{i}'] for i in range(4)])

  # specify the display parameters for the text
  text_parameters = {
      'text': 'label: {label}\ncirc: {circularity:.2f}',
      'size': 12,
      'color': 'green',
      'anchor': 'upper_left',
      'translation': [-3, 0],
  }

  # initialise viewer with coins image
  viewer = napari.view_image(image, name='coins', rgb=False)

  # add the labels
  label_layer = viewer.add_labels(label_image, name='segmentation')

  shapes_layer = viewer.add_shapes(
      bbox_rects,
      face_color='transparent',
      edge_color='green',
      properties=properties,
      text=text_parameters,
      name='bounding box',
  )

  napari.run()


    #                             img_ext_list, invert=False):

    # predictions = []
    # output_folder = []
    # return predictions, output_folder 