import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from glob import glob 
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from PIL import Image
import cv2
# Functions
def create_object_boxes(img_inpt, mask, Min_box_area, with_masks=True, data_tag='', annotation_directory='', disp=1):
    '''
    Create bouding boxes and segmentation masks for each detected fault
    img_inpt: input image
    mask: input mask
    Min_object_area : Minimal box area. smaller boxes will be ignored
    with_masks: create masks or not. False) only images for classification, True) mask for segmentation and object detection
    data_tag : a tag of the experiment/scan,etc
    annotation_directory: directory where the annoation will be save. 
                        if annotation_directory='', annotation will not be saved 
    '''
    if disp >=1:
        print(f'\n--> creating bouding boxes using Min box area = {Min_box_area}')
    if disp >=2:
        print(f'np.sum(mask)={np.sum(mask)} ')
        print(f'np.sum(img_inpt)={np.sum(img_inpt)} ')
        print(f'img_inpt type is {type(img_inpt)} ')
        print(f'img_inpt.shape={img_inpt.shape} ')
        print(f'Min_object_area={Min_box_area} ')
    #!python -m pip install -U scikit-image
    from skimage.measure import label, regionprops
    import cv2
    img_0 = img_inpt.copy()
    mask_0= mask.copy(); 
    mask_out=0.0*mask.copy(); 
    mask_0[mask>0]  = 1
    lbl_0 = label(mask_0, background=0, connectivity=2) 
    props = regionprops(lbl_0)
    img_box = img_0.copy(); img_box=250*(img_box/img_box.max())

    if disp>=1:
        display_detection(img_inpt, img_inpt, img_box, mask, msg=' input to create boxes', cmap="gray")
    try:
        img_box = cv2.cvtColor(img_box,cv2.COLOR_GRAY2RGB)
    except:
        img_box = img_array_gray2rgb(img_box)
    # Count boxes in the mask
    nb_box = 0
    label_boxes = []
    if disp>=1:
        display_detection(img_inpt, img_inpt, img_box, mask, msg=' input to create boxes [gray scale]', cmap="gray")
    if np.max(img_box)!=0:
        img_box= img_box / np.max(img_box)
        # mask_area = sum(sum(mask))
        mask_area = mask.shape[0] * mask.shape[1]
        defect_area =  np.count_nonzero(mask)
        if disp>=1:
            print(f'\n  - mask area = {mask_area}')
            print(f'\n  - found box countors ={len(props)} :\n {props}')
        if len(props)< mask_area/2 :
            for prop in props:
                box_obj= [prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]]
                box_area = (box_obj[2]-box_obj[0]) * (box_obj[3]-box_obj[1])
                if disp>=2:
                    print(f'\nprop={prop}')
                    print(f'box_area={box_area}')
                    print(f'Min_box_area={Min_box_area}')
                    print(f'mask shape={mask.shape}')
                    print(f'defect_area ={defect_area}')

                if box_area > Min_box_area**2 and Min_box_area>2:
                    label_boxes.append(box_obj)
                    if disp>=2:
                        print('Found bbox', prop.bbox)
                    cv2.rectangle(img_box, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), (255, 0, 0), 4)
                    nb_box+=1
                    # update th  mask
                    mask_out[prop.bbox[0]:prop.bbox[2],prop.bbox[1]:prop.bbox[3]] = mask[prop.bbox[0]:prop.bbox[2],prop.bbox[1]:prop.bbox[3]] 
                    if disp>=2:
                        display_detection(img_inpt, mask_0, img_box, mask_out, msg='Bouding boxes', cmap="gray")
        msg_org = ( int(0.1*mask.shape[1]) , int(0.95*mask.shape[0]) )
        if nb_box == 0:
            # print('\n\n\n Warnning: No fault is detected !!!!')
            img_box =cv2.putText(img=np.copy(img_box), text="Prediction: Good Object!", org=msg_org,fontFace=3, fontScale=1, color=(0,255,0), thickness=2)
            img_class = 'defect-free'
        else:
            img_box =cv2.putText(img=np.copy(img_box), text="Prediction: " + str(nb_box) + " faults detected!", org=msg_org,fontFace=3, fontScale=1, color=(255,0,0), thickness=2)
            img_class = 'defective'

        # save the annotaled/labeled data
        if annotation_directory!='':
            data_tag = data_tag +f'_box{Min_box_area}'
            out = save_xml_mask_annotations(img_inpt, mask_0, data_tag, annotation_directory,  label_boxes, img_class,  \
                                      CLASS_MAPPING = ['defect-free', 'defective'], defect_label ='', with_masks=with_masks, disp=0)
        # display 
        if disp>=2:
            msg_out= '- label_boxes=' + str(label_boxes) + \
                '\n- nb_box=' + str(nb_box) + \
                '\n- label_boxes=' + str(label_boxes)
            print(msg_out)

        return img_box, mask_out, nb_box, label_boxes, img_class
        
    else:
        print(f'boxes creation is ignored as the <img_box> is black (max value= {np.max(img_box)} )')
        return img_inpt, mask_out, nb_box, label_boxes, img_class

def save_volume(volume_arr, path):
    import nrrd
    nrrd.write(path, volume_arr)

def get_file_tag(path):
    TAG = os.path.basename(os.path.dirname(path)) + '--' + os.path.basename(path)
    return TAG

def save_xml_mask_annotations(img, mask, data_tag, annotation_directory,  label_boxes, img_class, with_masks=True,  \
    CLASS_MAPPING = ['defect-free', 'defective'], defect_label ='', img_ext = '.jpg', mask_ext = '.png', disp=0):
    '''
    Save annotation : bouding boxes in xml/csv files and save the binay mask images
    Min_box_area : the minimal detectable boxes area.
    dst_directory: directory where annotation will be saved in
    img: annotated images
    mask: annotation mask
    label_boxes: bouding boxes
    img_class: image class
    with_masks: create masks or not. False) only images for classification, True) mask for segmentation and object detection
    CLASS_MAPPING: list of all classes of this image. Default =  ['defect-free', 'defective']
    defect_label: list defining the label of each box in  <label_boxes> [ missing screw, missing spring, ...] , default:''

    '''

    if with_masks == False:
        # adjust the the destination folder
        annotation_directory=os.path.join(annotation_directory, img_class, os.path.dirname(data_tag) )
        data_tag = os.path.basename(data_tag)
        # save images and mask in RGB files
        if disp >=1:
            print(f'\n ---> saving the annotation into the folder [{annotation_directory}]')
        image_path = os.path.join( annotation_directory, data_tag + img_ext)
        save_array_to_2rgb(img, image_path)
        msg_out=  '\n- The {img_class} images are saved into the folder [{dst_directory}]' 
        if disp >=2:
            print(msg_out)
        return msg_out

    else:
        # initializations
        CSV_DATA_FILE = os.path.join(annotation_directory, 'data.csv')
        xml_list = []
        # adjust the the destination folder
        annotation_directory=os.path.join(annotation_directory, img_class, os.path.dirname(data_tag) )
        data_tag = os.path.basename(data_tag)

        # save images and mask in RGB files
        if disp >=1:
            print(f'\n ---> saving the annotation into the folder [{annotation_directory}]')
        image_path = os.path.join( annotation_directory, 'images', data_tag + img_ext)
        save_array_to_2rgb(img, image_path)
        mask_path= os.path.join( annotation_directory, 'masks', data_tag  + mask_ext)  
        save_array_to_gray(mask, mask_path)
        # transform the inspection boxes to xml annotations
        w, h = img.shape[1], img.shape[0]   # get the image dimensions
        if  defect_label=='':            # Assign the default name of defectes as  CLASS_MAPPING[1]->'defective'                                     
            defect_label = [CLASS_MAPPING[1] for defect in label_boxes] 
        xml_list = get_box_coordinate(image_path, label_boxes, w, h, defect_label, xml_list=xml_list)
        # Save  all annotation in csv file
        column_name = ['filename', 'image_id', 'width', 'height', 'bbox', 'inspection']
        if not os.path.exists(CSV_DATA_FILE):
            xml_df = pd.DataFrame(xml_list, columns=column_name)
        else:
            xml_old = pd.read_csv(CSV_DATA_FILE)
            xml_new = pd.DataFrame(xml_list, columns=column_name)
            xml_df = pd.concat([xml_old, xml_new])

        # save the annotation
        xml_df.to_csv(CSV_DATA_FILE,index=None)
        if disp >=1:
            print('\n xml_list=', xml_list)
            print('\n CSV file is saved in : %s'%(CSV_DATA_FILE))
            print('xml_df', xml_df )

        # save classes in jason files
        import json
        DIR_CLASS = os.path.join( os.path.dirname(CSV_DATA_FILE), 'classes.json')
        with open(DIR_CLASS, 'w') as fp:
            json.dump(CLASS_MAPPING, fp)
        # display
        msg_out=  '\n- The annotation are saved into the folder [{dst_directory}]' 
        if disp >=2:
            print(msg_out)
        return msg_out

def get_box_coordinate(image_path, detection_boxes, w, h, defect_class, xml_list=[], disp=0):
    '''
    get labels/masks/boxes coordinates from detections 
    '''
    import os
    if disp>=1:
        print("\n\n\n ######\n Image = %s \ndetection = %s"%(image_path, detection_boxes) )
    # get file ID / prefix
    file_prefix, file_extension = os.path.splitext(os.path.basename(image_path) ) 
    voc_labels = []
    for box, defect_name in zip(detection_boxes, defect_class):	
        voc = []
        voc.append(defect_name)
        # bbox_width = float(data[3]) * w
        # bbox_height = float(data[4]) * h
        # center_x = float(data[1]) * w
        # center_y = float(data[2]) * h
        if disp>=1:
            print('defect_name :', defect_name)
            print('box :', box)

        # create the coordinate 
        bbox_width = float(box[2]) * w
        bbox_height = float(box[3]) * h
        center_x = float(box[0]) * w
        center_y = float(box[1]) * h
        voc.append(center_x - (bbox_width / 2))
        voc.append(center_y - (bbox_height / 2))
        voc.append(center_x + (bbox_width / 2))
        voc.append(center_y + (bbox_height / 2))
        voc_labels.append(voc)
        # print(' voc_labels :', voc_labels)
        column_name = ['filename', 'image_id', 'width', 'height', 'bbox', 'source']
        row=[]
        row.append(image_path)
        row.append(file_prefix)
        row.append(w)
        row.append(h)
        # position = str(data[2:6])
        position = [ str(b) for b in box]
        # print('position=', position)
        row.append( position  )
        row.append(defect_name)
        xml_list.append(row)

    # print
    # print('voc_labels=', voc_labels)# save XML
    image_path = image_path.replace('\\', '/')
    file_path = image_path.replace('images/', 'annotations/')
    dst_xml_dir = os.path.dirname(file_path); create_new_directory(dst_xml_dir)
    xml_filename = os.path.join(dst_xml_dir, file_prefix + '.xml')
    create_file(dst_xml_dir, file_prefix, xml_filename, w, h, voc_labels)
    # report
    if disp>=1:
        print("\n\nThe annotation of image: {} is complete. \nThe labels are stored in {}".format(image_path, xml_filename))

    return xml_list

import os
import xml.etree.cElementTree as ET
from PIL import Image

def create_file(root_dir, file_prefix, xml_filename,  width, height, voc_labels):
    root = create_root(root_dir, file_prefix, width, height)
    root = create_object_annotation(root, voc_labels)
    tree = ET.ElementTree(root)
    
    # print('xml_filename=', xml_filename)
    tree.write(xml_filename)

def create_root(root_dir, file_prefix, width, height, ext='.jpg'):
    root = ET.Element("annotations")
    ET.SubElement(root, "filename").text = file_prefix + ext
    ET.SubElement(root, "folder").text = root_dir
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    return root

def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])
    return root

def get_class_name(idx):
  return '"' + str(idx) + '"'

def display_detection(img_ref, img_input, img_box, mask,  msg='', cmap="cividis"):
    print(f'\n\n --> .\n- size:  Ref={img_ref.shape},  \
            input={img_input.shape},  output={img_box.shape}')
    if len(img_box.shape)==3 and img_box.shape[2]!=3:
        import random 
        idx = random.randint(1,img_box.shape[0])
        img_box = img_box[idx]
    # display
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, sharey=True)
    ax1.imshow(img_input, interpolation='none', cmap=cmap)
    ax1.set_title('Input image')
    ax1.set_ylabel(msg)
    ax2.imshow(img_ref, interpolation='none', cmap=cmap)
    ax2.set_title('Registred reference image')
    ax3.imshow(img_box, interpolation='none', cmap=cmap)
    ax3.set_title('Defect localization ')
    ax4.imshow(img_input, cmap='gray', interpolation='none', origin='lower')
    ax4.imshow(mask, cmap='Reds', interpolation='none', alpha=0.6)
    ax4.set_title(f'Defect segmentation ')
    plt.show()

def check_supported_data_paths(path):
    file_type = os.path.isfile(path)
    if file_type:
        # Draw rek rek filesL
        filename, file_extension = os.path.splitext(path)
        if file_extension == '.rek' or file_extension == '.bd' or file_extension == '.nii' or file_extension == '.nrrd':
            return '3d'
        elif file_extension == '.tif' or file_extension == '.tiff' or file_extension == '.jpg' or file_extension == '.png':
            return '2d'
        else:
            msg = '\n Warning: The file format : %s is not supported!!!!'%(path)
            raise Exception(msg)

    else:
        print('Error: the input path [{path}] is not a file!!!!')
        return '-1'

def get_file_tag(path):
    TAG = os.path.basename(os.path.dirname(path)) + '--' + os.path.basename(path)
    return TAG

def get_folder_tag(path):
    TAG = os.path.basename(os.path.dirname(os.path.dirname(path))) + '--' + os.path.basename(os.path.dirname(path))
    return TAG

def load_image(path):
    import numpy as np
    import cv2
    # print(' The selected image is :', path)
    filename, file_extension = os.path.splitext(path)
    try:
        img = cv2.imread(path,0)
    except:
        msg = '\n Error: the image path ' + path + 'cannot be loaded!!!!'
        raise Exception(msg)
    return img

def load_data(path):
    data_type_ref = check_supported_data_paths(path)
    if data_type_ref=='3d':
        ref_volume_arr=load_volume(path)
        print(f' The selected volume file is [{path}] of size {ref_volume_arr.shape}')
        return ref_volume_arr

    elif data_type_ref=='2d':
        print(' The selected image file is :', path)
        ref_image_arr=load_image(path)
        print(f' The selected image file is [{path}] of size {ref_image_arr.shape}')
        return ref_image_arr

    else:
        msg = f'\n\n the input file format [{path}] is not supported !!!'
        raise Exception(msg)

def load_raw_data(path):
    if os.path.exists(path):
        if os.path.isdir(path):
            ref_volume_arr=load_slices_to_volume(path)
            return ref_volume_arr
        elif os.path.isfile(path):
            data = load_data(path)
            return data
    else:
        print(path)
        msg = f'\n\nError: the path [{path}] cannot be found!!!!\nplease recheck the configuration file config/config.yml\n\n'
        raise Exception(msg)

def load_input_data_path_or_array(file):
    try: 
        file.shape
        # Assign the array 
        arr = np.copy(file)
    except:
        # load path file
        arr = load_raw_data(file)
    return arr

def load_slices_to_volume(root, ext='.tiff'):
    from glob import glob 
    try : 
        os.path.isdir(root)
        list_slices_paths = glob(os.path.join(root, '*' + ext ))
    except:
        list_slices_paths = root
    
    return read_concatenate_slices_(list_slices_paths)

def read_concatenate_slices_(list_slices_paths):
    from PIL import Image
    import numpy as np
    slices = []
    img_arr=[]
    # concatenate the slices (axix0)   
    for img_path in list_slices_paths:
        img = Image.open(img_path)
        slices.append(img)
        img_arr.append(np.array(img))
    #stack the 2D arrays over one another to generate the 3D array
    volume_array=np.stack(img_arr, axis=1)
    # print(volume_array.shape)     
    return  volume_array

def VGI2Py(file_path):
    import os
    # Using readlines()
    vgi_file =file_path + '.vgi'
    if not os.path.exists(vgi_file):
        print(' Error: The VGI file is not found: \n', vgi_file)

        return 0
    else:
        file1 = open(vgi_file, 'r')
        Lines = file1.readlines()
        count = 0
        # Strips the newline character
        for line in Lines:
            count += 1
            line_str= line.strip()
            terms = line_str.split(' ')
            # print("Line{}: {} \n {}".format(count, line_str, terms ))

            if terms[0]=='size':
                size=(int(terms[2]), int(terms[3]), int(terms[4]))
                print(' size = ', size)
            elif terms[0]=='bitsperelement':
                if terms[2]=='8':
                    voxel_type = np.uint8

                elif terms[2]=='16':
                    voxel_type = np.uint16

                else:
                    print(' Voxel type is not an usual value = ', terms[2])

            
                print(' voxel_type = ', voxel_type)

            elif terms[0]=='SkipHeader':
                SkipHeader=int(terms[2])
                print(' SkipHeader = ', SkipHeader)
            
        # load the BCM volume
        voxel_count = size[0] * size[1] * size[2]
        f = open(file_path,'rb') #only opens the file for reading
        vol_arr=np.fromfile(f,dtype=voxel_type,offset=SkipHeader,count=voxel_count)
        f.close()
        vol_arr=vol_arr.reshape(size[0],size[1],size[2])
        return vol_arr

def simpleRek2Py(filename, image_width, image_height, image_depth, voxel_datatype):
    
    '''
    filename: path to the rek file
    
    image_width x image_height x image_depth: the dimension to be resized
    500x500x500 - 0.2 GB file
    1000x1000x1000 - 1.4 GB file
    2000x2000x2000 - 15.5 GB file

    voxel_datatype: the datatype of the file
    uint16 - integer data file
    float32 - float data file
    '''
    print('\n Opening rek file: %s\n  - size=(%d,%d,%d) \n  - voxel_datatype=%s '%(filename, image_width, image_height, image_depth,voxel_datatype))
    
    if (voxel_datatype == "uint16"):
        datatype = np.uint16
    elif (voxel_datatype == "float32"):
        datatype = np.float32
    else:
        raise ValueError("Unsupported datatype")

    with open(filename, 'rb') as fd:
        raw_file_data = fd.read()        
    image = np.frombuffer(raw_file_data[2048:], dtype=datatype)
    shape = image_width, image_height, image_depth

    return image.reshape(shape)

def load_volume( path):
        import numpy as np
        # Draw rek rek filesL
        filename, file_extension = os.path.splitext(path)
        if os.path.exists(path):
            # read 3D volume from rek file
            if file_extension=='.rek':
                # volume_array = scanner.load_Rek2Py(rek_file)
                # enter slice dimensions (width x height x depth)
                image_width = 500
                image_height = 500
                image_depth = 500
                # enter voxel datatype ("float32" or "uint16")
                voxel_datatype = "uint16"
                # read 3D volume from rek file
                
                volume_array = simpleRek2Py(path, image_width, image_height, image_depth, voxel_datatype)  
            
            elif file_extension=='.bd': 

                volume_array = VGI2Py(path)
  

            elif file_extension=='.nii':
                import nibabel as nib
                volume_array = nib.load(path).get_data()


            elif file_extension=='.nrrd':
                import nrrd
                volume_array, _ = nrrd.read(path)

                
            else: 
                msg = '\n Warning: The file format : %s is not supported!!!!'%(path)
                raise ValueError(msg)

        else:
            msg = '\n Warning: The file : %s is not found!!!!'%(path)
            raise Exception(msg)
        # sanity check the volume mush have this shape [N, M, N]
        if len(np.unique(volume_array.shape))<=2:
            if volume_array.shape[0]!=volume_array.shape[2]:
                volume_array = volume_array.transpose(0, 2, 1)
        else: 
            print('Error: the volume does not respect the shape critereon [N, M, N]!')
            return -1

        return volume_array

def create_volume_from2D(folder_path,target_path,volume_name):
    import nrrd
    from PIL import Image
    slices = []
    img_arr=[]
    img_count=1
    for f in os.listdir(folder_path):
        img = Image.open(folder_path+'/'+f)
        slices.append(img)
        img_arr.append(np.array(img))
        print(f'[+] Slice read: {img_count}', end='\r')
        img_count=1+img_count
    print()
    #stack the 2D arrays over one another to generate the 3D array
    print('creating volume')
    volume_array=np.stack(img_arr, axis=1)
    print(volume_array.shape)
    print('saving volume')
    nrrd.write(target_path+'/'+volume_name+'.nrrd',volume_array)
    print('volume saved')

def load_tif_as_jpg(img_path, scale=0.0, save=False):
    from PIL import Image
    cnt = 0
    target_name ='data/temp.jpg'
    if os.path.isfile(img_path):
        img = Image.open(img_path)
        img=img.point(lambda i:i*(1./256)).convert('L')
    else:
        # convert grey image, to 3-channel RGB
        result = img_array_gray2rgb(img_path, scale=scale)
        # Save result
        img = Image.fromarray(result)
    create_new_directory(os.path.dirname(target_name))
    img.save(target_name, "JPEG")

    return  cv2.imread(target_name)

def img_array_gray2rgb(array_gray, scale=255):
    # convert grey image, to 3-channel RGB
    img_ = np.copy(array_gray)
    # normaliztion and scaling
    if scale > 1:
        img_= img_ /np.max(img_)
        img_ = img_*scale
    img_rgb = np.zeros((*img_.shape,3), dtype=np.uint8)
    for k in range(3):
        img_rgb[:,:,k]=img_
    return img_rgb

def save_array_to_gray(arr, filename, scale=255):
    '''
    save a 2D array to png grayscale image 
    arr : input 2D numpy array
    filename : input image file name
    scale : save image in a different scale. Default [0, 255]
    '''
    if len(arr.shape) == 2:
        create_new_directory(os.path.dirname(filename) )
        image_int = np.array(255*arr, np.uint8)
        cv2.imwrite(filename, image_int)
    else:
        msg = f'\n\n--> Error: the input array dimension [{arr.shape}] is not a 2D image!!!'
        raise Exception(msg)

def create_new_directory(DIR):
  if not os.path.exists(DIR):
    os.makedirs(DIR)

def save_array_to_2rgb(arr, filename, scale=255):
    '''
    save a 2D array to jpg image in RGB format
    arr : input 2D numpy array
    filename : ipg image file name
    scale : save image in a different scale. Default [0, 255]
    '''
    if len(arr.shape) == 2:
        create_new_directory(os.path.dirname(filename) )
        img_rgb = img_array_gray2rgb(arr, scale=scale)
        img = Image.fromarray(img_rgb)
        img.save(filename, "JPEG")
    else:
        msg = f'\n\n--> Error: the input array dimension [{arr.shape}] is not a 2D image!!!'
        raise Exception(msg)

def convert_tif_jpg(list_images):
    from PIL import Image

    print('\n--------------------------------------------')
    print('\nConversion in process ...!\nPlease wait :):)')
#     print('list_images=', list_images)
    cnt = 0
    for file in list_images:
#         print(file)
        if os.path.exists(file):
            file_=file.split("\\")[-1]
            filename=file_.split(".")
            target_name = '/'.join(file.split("\\")[:-1]) + '/' +  filename[0] + ".jpg"
            if cnt%100==0:
                print('target_name%d = %s '%(cnt, target_name) )

            img = Image.open(file)
            img.mode = 'I'
            img.point(lambda i:i*(1./256)).convert('L').save(target_name)

            # rgb_image = img.convert('RGB')
            # rgb_image.save(target_name)
#             print("Converted image saved as " + target_name)
            cnt = cnt +1
        else:
            print(file + " not found in given location")
    print('\n\nConversion done!\n%d images were converted to jpg'%(cnt))

def load_volume_and_annotation_mask(vol_path):
    # load volume or mask
    try:
        vol = load_volume(vol_path)
    except:
        import tifffile
        vol = tifffile.imread(vol_path)
    return vol
#____________________________________________________________________________---
def Export_annotations_from_folders(args):
    # input data
    root = args.root                      # full path of the flder  where the folders with volume/masks are saved
    vol_tag=args.vol_tag                  # the tag/name of the input volume data
    mask_tag=args.mask_tag                # the tag/name of the mask volume data
    annotation_directory = args.dst       # full path of the folder where the annotation will be saved
    BoxMin = args.BoxMin                  # Minimal box area. smaller boxes will be ignored. defaut = 10
    with_masks = args.with_masks          # enable the creatation of mask/annotation. defaut = True
    sep_folders = args.sep_folders        # enable annotation in seperate folder. defaut = False
    subfolder_list =  glob(os.path.join(root,'*',''), recursive=True) 
    # Display
    print('###################################################################################')
    print(f'#              Exporting annotation using Inspection results the model           ')
    print(f'#  root={root} \n#  annotation directory={annotation_directory}')
    print(f'#  BoxMin={BoxMin},  with_masks={with_masks},  sep_folders={sep_folders}')
    print('###################################################################################')
    print(f'--> found {len(subfolder_list)}data folderes: \n{subfolder_list} ')

    # run
    for subfolder in subfolder_list:
        print(f'\n --> Checking the folder : {subfolder}')
        volume_path = glob(subfolder+'*'+vol_tag) 
        mask_path = glob(subfolder+'*'+mask_tag )
        # napari mask 
        if len(mask_path)==0:
            mask_path = glob(subfolder+'*.tif' )
        # process the found volume/masks
        if len(volume_path)==len(mask_path)==1:
            print(f'\n volume = {volume_path} \n mask = {mask_path}')
            # save annotations 
            data_tag = get_folder_tag(volume_path[0])
            err = Export_annotation_from_mask(volume_path[0], mask_path[0], Min_box_area=BoxMin, data_tag=data_tag, \
                                            annotation_directory=annotation_directory, with_masks=with_masks, sep_folders=sep_folders, disp=1)

def Export_annotation_from_mask(volume_path, mask_path, Min_box_area, data_tag='', annotation_directory='', with_masks=False, sep_folders=True, disp=0):
    '''
    A createanootation with or without masks for each detected fault
    volume_path: input volume
    mask: input mask
    Min_object_area : Minimal box area. smaller boxes will be ignored
    with_masks: create masks or not. False) only images for classification, True) mask for segmentation and object detection
    sep_folders : enable annotation in seperate folder, Default=True
    data_tag : a tag of the experiment/scan,etc
    annotation_directory: directory where the annoation will be save. 
                        if annotation_directory='', annotation will not be saved 
    '''
    vol = load_volume_and_annotation_mask(volume_path)
    mask_vol = load_volume_and_annotation_mask(mask_path)
    if mask_vol.shape != vol.shape:
        print(f'Error: the mask and volume dont have the same dimensions: \n - volume={vol.shape}\n - mask={mask_vol.shape} ')
        return 1
    else:

        if disp>=1:
            print(f'\n --> annotate the {vol.shape[1]} slices and annotations ... ')
        dst_folder = annotation_directory# os.path.join(annotation_directory , data_tag)
        for idx in range(vol.shape[1]):
            # defect annotation with bouding boxes and masks
            if annotation_directory!='':
                if sep_folders :
                    data_tag_= os.path.join(data_tag , data_tag+f'_slice{idx}')
                else:
                    data_tag_= data_tag+f'_slice{idx}'
                img_box, mask_out, nb_box, label_boxes, img_class = create_object_boxes(vol[:,idx,:], mask_vol[:,idx,:], Min_box_area=Min_box_area, disp=disp-1, \
                                                                data_tag=data_tag_, annotation_directory=dst_folder, with_masks=with_masks)

        return 0

def segment_teeth(image, gray):
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    from scipy import ndimage
    import argparse
    import imutils
    import cv2

    # shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    # cv2.imshow("Input", image)

    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    # gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    plt.imshow(thresh); plt.title('thresh')

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    print("[INFO] {} unique contours found".format(len(cnts)))

    # loop over the contours
    for (i, c) in enumerate(cnts):
        # draw the contour
        ((x, y), _) = cv2.minEnclosingCircle(c)
        cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    image = cv2.line(image, (100,100), (300,300), (0,0,255),4)
    # show the output image
    plt.imshow(image); plt.title('Image')
    plt.show()
    

def create_annotation_from_mask():
    image_path = 'D:/Datasets/images-dataset/labeled-data/segmentation-dataset/Tufts_Dental_Database/images/28.jpg'
    mask_path = 'D:/Datasets/images-dataset/labeled-data/segmentation-dataset/Tufts_Dental_Database/masks/28.jpg'
    Min_box_area = 10
    image = load_image(image_path)
    mask = load_image(mask_path)

    segment_teeth(image, mask)

    # image_out, mask_out, nb_box, label_boxes, img_class = create_object_boxes(image, mask, Min_box_area, with_masks=True, \
    #                                                              data_tag='', annotation_directory='', disp=1)

    
    # # Using the Canny filter to get contours
    # edges = cv2.Canny(mask, 20, 30)
    # # Using the Canny filter with different parameters
    # edges_high_thresh = cv2.Canny(mask, 60, 120)
    # # Stacking the images to print them together
    # # For comparison
    # images = np.hstack((mask, edges, edges_high_thresh))

    # # Display the resulting frame
    # plt.imshow(images)
    # plt.title('edges')
    # plt.show()
 
def prepare_parser():
    parser = ArgumentParser(description='Inspection-based annotation')
    parser.add_argument(
        "-root",
        required=True,
        metavar="DIRECTORY",
        help="full path of the flder  where the folders with volume/masks are saved",
        type=str,
    )
    parser.add_argument(
        "-vol_tag",
        required=True,
        default='volume.nrrd',
        metavar="string TAG",
        help="the tag/name of the input volume data",
        type=str,
    )
    parser.add_argument(
        "-mask_tag",
        required=True,
        default='mask.nrrd',
        metavar="string TAG",
        help="the tag/name of the mask volume data",
        type=str,
    )
    parser.add_argument(
        "-dst",
        required=True,
        metavar="DIRECTORY",
        help="full path of the folder where the annotation will be saved",
        type=str,
    )
    parser.add_argument(
        "-BoxMin",
        default=10,
        metavar="DIRECTORY",
        help="Minimal box area. smaller boxes will be ignored. defaut = 10",
        type=int,
    )
    parser.add_argument(
        "-with_masks",
        default=True,
        metavar="bool",
        help="enable the creatation of mask/annotation. defaut = True",
        type=bool,
    )
    parser.add_argument(
        "-sep_folders",
        default=False,
        metavar="bool",
        help="enable annotation in seperate folder. defaut = False",
        type=bool,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

# python -root   -vol_tag  -mask_tag  -dst
def main():
    parser = prepare_parser()
    args =  parser.parse_args()
    Export_annotations_from_folders(args)
if __name__ == '__main__':
    # main()
    create_annotation_from_mask()
