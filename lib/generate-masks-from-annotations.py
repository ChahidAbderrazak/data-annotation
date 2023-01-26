import os, gc
import cv2
import sys
import json
import time
import signal
import numpy as np 
import pandas as pd
from tqdm import tqdm
import imutils
import matplotlib.pyplot as plt

# input variables

# # --- HAIS_DATABASE-medium-speed
# source_folder = "raw-dataset/HAIS_DATABASE-medium-speed__CSI_CAMERA"
# json_path = os.path.join(source_folder,"annotations/annotations 346 of 1800.json" )	#"maskGen_json.json"										 # Relative to root directory
# dst_folder='/media/abdo2020/DATA1/data/labeled-dataset/segmentation-dataset'

# --- Tejush manual annotation
source_folder = '/media/abdo2020/DATA1/data/labeled-dataset/HAIS-project/manual-annotation/set 1 - 526'
json_path = os.path.join(source_folder,"annotations/annotations.json" )	#"maskGen_json.json"										 # Relative to root directory
dst_folder='/media/abdo2020/DATA1/data/labeled-dataset/segmentation-dataset/tmp'


disp=0
#---------------------------------------------------------------------
# retreive the folders and annotations 
dst_folder=os.path.join(dst_folder, os.path.basename(source_folder))

def create_new_directory(DIR):
	if not os.path.exists(DIR):
		os.makedirs(DIR)

# exiting the code by Ctrl-C click
def handler(signum, frame):
	exit(1)
signal.signal(signal.SIGINT, handler)

# Create the dataset folders
image_folder = os.path.join(dst_folder, "images")
mask_folder = os.path.join(dst_folder, "masks")
mask_color_folder = os.path.join(dst_folder, "masks_color")
create_new_directory(image_folder)
create_new_directory(mask_folder)
create_new_directory(mask_color_folder)
# start dataset generation 
count = 0																						# Count of total annotated objects saved
cnt_img=0																						# Count of total images/masks saved
file_bbs = {}																				# Dictionary containing polygon coordinates for mask
list_classes=[]
# Read JSON file
try:
	with open(json_path) as f:
		data = json.load(f)
except Exception as e:
	print(f'\n Error in readind the annoation file!!! \nException: {e}')
	sys.exit(0)
#%%#################### Mask generation	###########################
# Extract X and Y coordinates if available and update dictionary
def add_to_dict(data, itr, key, count):
		try:
			x_points = data[itr]["regions"][count]["shape_attributes"]["all_points_x"]
			y_points = data[itr]["regions"][count]["shape_attributes"]["all_points_y"]
			label=data[itr]["regions"][count]["region_attributes"]["Type"]
		except Exception as e:
			if disp>=1:
				print("No BB. Skipping", key)
			return
		
		all_points = []
		for i, x in enumerate(x_points):
				all_points.append([x, y_points[i]])
		if not label in list_classes:
			list_classes.append(label)
		file_bbs[key] = {'all_points': all_points, 'classe':label}

def save_classes_json(class_file, classes_list):
	import json
	if os.path.exists(class_file):
		return load_classes(class_file)
	else:
		if len(classes_list)<255:
			dict_classes={clss:255-k for k, clss in enumerate(classes_list)}
		else:
				dict_classes={clss:k+1 for k, clss in enumerate(classes_list)}
		# update background
		# print(f'\n dict_classes={dict_classes}, condition={ 0 in classes_list}')
		if not 0 in classes_list:
			dict_classes.update({"Background": 0})
		
		# save the classes
		save_json(class_file, dict_classes)
		return dict_classes

def load_classes(class_file):
	import json
	if os.path.exists(class_file):
			with open(class_file) as json_file:
					dict_classes = json.load(json_file)
	else:
			msg=f'\n\n Error: the class JSON file ({class_file}) does not exist!!!'
			raise Exception(msg)
	return dict_classes

def save_json(class_file, dict_classes):
		# create the folder
		create_new_directory(os.path.dirname(class_file))
		# save the classe JSON file
		with open(class_file, 'w') as outfile:
				json.dump(dict_classes, outfile,indent=2)

def get_label_name(dict_classes, id):
	if id==0:
		return 'Background'
	for name in list(dict_classes.keys()):
		if dict_classes[name]==id:
			return name

	return 'Undefined'

def color_the_mask(mask, colors):
	seg=np.unique(mask)
	# print(f'flag: \n - mask seg={seg} \n - colors={colors}')
	for k, pixel in enumerate(seg):
		if pixel!=0:
			color_mask=np.zeros_like(mask)+ colors[str(pixel)]
			mask[mask==pixel]=color_mask[mask==pixel]

def update_object_name(arr, labels_names, MASK_WIDTH, MASK_HEIGHT):
		thresh = np.zeros((MASK_WIDTH, MASK_HEIGHT,3), dtype=np.uint8)
		cv2.fillPoly(thresh, [arr], color=(label, label, label))
		# gray = cv2.cvtColor(thresh,cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(thresh, 0, 2*label)
		edges = cv2.dilate(edges, None)
		edges = cv2.erode(edges, None)
		#-- Find contours in edges, sort by area ---------------------------------------------
		cnts= cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		cnts = imutils.grab_contours(cnts)
		# loop over the contours
		for c in cnts:
			# compute the center of the contour
			M = cv2.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			labels_names.append([classe,(cX - 20, cY - 20)])

#%%#################### Bounding boxes generation	###########################

def create_object_boxes(image_path, mask, dict_classes, annotation_directory='dataset', disp=1): 
	'''
	Create bounding boxes and segmentation masks for each detected fault
	mask: input mask
	dict_classes : classes of the mask
	annotation_directory: directory where the annoation will be save. 
											if annotation_directory='', annotation will not be saved 
	'''
	## get the boxes from mask
	from skimage.measure import label, regionprops
	import cv2
	# Count boxes in the mask
	nb_box=0
	label_boxes=[]
	defect_label=[]
	# annotate the detected bounding boxes
	segments=[ l for l in np.unique(mask) if l!=0] 
	if disp>=1:
		print(f'\n - the mask [shape={mask.shape}] contains {len(segments)} lables={segments}')
	if np.max(mask)!=0: # the mask contains defects
		for label_pixel in segments:
			mask_0=mask.copy(); 
			mask_0[mask==label_pixel]=1
			lbl_0=label(mask_0, background=0, connectivity=2) 
			props=regionprops(lbl_0)

			# count the	number of detected boxes above the Threshold
			for prop in props: 
					box_obj=[prop.bbox[1], prop.bbox[0], prop.bbox[3], prop.bbox[2]]
					y,x=int( 0.5*(box_obj[0]+box_obj[2]) ), int( 0.5*(box_obj[1]+box_obj[3]) )
					label_=get_label_name(dict_classes, label_pixel)
					label_boxes.append(box_obj)
					defect_label.append(label_)
					nb_box+=1
					if disp>=2:
						mask0=mask.copy()
						print(f'\n label position={(x,y)} --> id={id}')
						cv2.rectangle(mask0, (box_obj[0], box_obj[1]), (box_obj[2], box_obj[3]), (255), 4)
						plt.imshow(mask0)
						plt.title(f'object={label_}')
						plt.show()
			if annotation_directory!='': 
				out=save_xml_mask_annotations(image_path, mask_0, annotation_directory, label_boxes, defect_label, \
																	dict_classes=dict_classes, disp=0)
			# display 
			if disp>=1: 
					msg_out='- label_boxes=' + str(label_boxes) + \
							'\n- nb_box=' + str(nb_box) + \
							'\n- label_boxes=' + str(label_boxes) + out
					print(msg_out)
			return nb_box, label_boxes
			
	else: 
		print(f'boxes creation is ignored as the <mask> is black (max value={np.max(mask)} )')
		return nb_box, label_boxes

def save_xml_mask_annotations(image_path, mask, annotation_directory, label_boxes, defect_label, dict_classes, disp=0): 
	'''
	annotation_directory: directory where annotation will be saved in
	mask: annotation mask
	label_boxes: bounding boxes
	CLASS_MAPPING: list of all classes of this image. Default=['defect-free', 'defective']
	defect_label: list defining the label of each box in	<label_boxes> [ missing screw, missing spring, ...], default: ''

	'''
	# initializations
	CSV_DATA_FILE=os.path.join(annotation_directory, 'data.csv')
	xml_list=[]
	
	# save images and mask in RGB files
	if disp >=1: 
		print(f'\n ---> saving the annotation into the folder [{annotation_directory}]')
	
	# transform the inspection boxes to xml annotations
	w, h=mask.shape[1], mask.shape[0]	 # get the image dimensions
	xml_list=get_box_coordinate(image_path, label_boxes, w, h, defect_label, xml_list=xml_list)
	
	# Save	all annotation in csv file
	column_name=['filename', 'image_id', 'width', 'height', 'bbox', 'inspection']
	if not os.path.exists(CSV_DATA_FILE): 
			xml_df=pd.DataFrame(xml_list, columns=column_name)
	else: 
			xml_old=pd.read_csv(CSV_DATA_FILE)
			xml_new=pd.DataFrame(xml_list, columns=column_name)
			xml_df=pd.concat([xml_old, xml_new])

	# save the annotation
	xml_df.to_csv(CSV_DATA_FILE, index=None)
	if disp >=1: 
			print('\n xml_list=', xml_list)
			print('\n CSV file is saved in : %s'%(CSV_DATA_FILE))
			print('xml_df', xml_df )
	msg_out=''
	return msg_out
def get_box_coordinate(image_path, detection_boxes, w, h, defect_class, xml_list=[], disp=0): 
		'''
		get labels/masks/boxes coordinates from detections 
		'''
		import os
		if disp>=1: 
				print("\n\n\n ######\n Image=%s \ndetection=%s"%(image_path, detection_boxes) )
		# get file ID / prefix
		file_prefix, file_extension=os.path.splitext(os.path.basename(image_path) ) 
		voc_labels=[]
		for pascal_voc_box, defect_name in zip(detection_boxes, defect_class): 	
				voc_labels.append([defect_name] + list(pascal_voc_box) )
				if disp>=1: 
						print('defect_name : ', defect_name)
						print('pascal_voc_box : ', pascal_voc_box)
				yolo_box=pascal_voc_to_yolo(pascal_voc_box[0], pascal_voc_box[1], pascal_voc_box[2], pascal_voc_box[3], w, h)
				column_name=['filename', 'image_id', 'width', 'height', 'bbox', 'source']
				row=[image_path, file_prefix, w, h, yolo_box, defect_name]
				xml_list.append(row)
		# copy the image
		image_path=image_path.replace('\\', '/')
		file_path=image_path.replace('images/', 'annotations/')
		dst_xml_dir=os.path.dirname(file_path)
		# input(f'\n dst_xml_dir={dst_xml_dir}')

		create_new_directory(dst_xml_dir)
		xml_filename=os.path.join(dst_xml_dir, file_prefix + '.xml')

		# create the xml file
		# input(f'\n voc_labels={voc_labels} \n xml_filename={xml_filename}')
		create_file(dst_xml_dir, file_prefix, xml_filename, w, h, voc_labels)

		# report
		if disp>=1: 
				print(f"\n\nThe annotation of image is complete! \n - image file: {image_path}	\n - xml file: {xml_filename}")

		return xml_list

def get_class_name(idx): 
	return '"' + str(idx) + '"'

import os
import xml.etree.cElementTree as ET
from PIL import Image

def create_file(root_dir, file_prefix, xml_filename, width, height, voc_labels): 
		# input(f'\n xml_filename={xml_filename}, size={[width, height]}, voc_labels={voc_labels}')
		root=create_root(root_dir, file_prefix, width, height)
		root=create_object_annotation(root, voc_labels)
		tree=ET.ElementTree(root)
		
		tree.write(xml_filename)
		
def create_root(root_dir, file_prefix, width, height, ext='.jpg'): 
		root=ET.Element("annotations")
		ET.SubElement(root, "filename").text=file_prefix + ext
		ET.SubElement(root, "folder").text=root_dir
		size=ET.SubElement(root, "size")
		ET.SubElement(size, "width").text=str(width)
		ET.SubElement(size, "height").text=str(height)
		ET.SubElement(size, "depth").text="3"
		return root

def create_object_annotation(root, voc_labels): 
		for voc_label in voc_labels: 
				obj=ET.SubElement(root, "object")
				ET.SubElement(obj, "name").text=voc_label[0]
				bbox=ET.SubElement(obj, "bndbox")
				ET.SubElement(bbox, "xmin").text=str(voc_label[1])
				ET.SubElement(bbox, "ymin").text=str(voc_label[2])
				ET.SubElement(bbox, "xmax").text=str(voc_label[3])
				ET.SubElement(bbox, "ymax").text=str(voc_label[4])
		return root
#%%######################	 BOUNDING BOXES CONVERTIONS	 ######################

# source: : https: //christianbernecker.medium.com/convert-bounding-boxes-from-coco-to-pascal-voc-to-yolo-and-back-660dc6178742
# Converting Coco 
def coco_to_pascal_voc(x1, y1, w, h): 
		return [x1, y1, x1 + w, y1 + h]

def coco_to_yolo(x1, y1, w, h, image_w, image_h): 
		return [((2*x1 + w)/(2*image_w)), ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]

# Converting Pascal_voc
def pascal_voc_to_coco(x1, y1, x2, y2): 
		return [x1, y1, x2 - x1, y2 - y1]

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h): 
		return [((x2 + x1)/(2*image_w)), ((y2 + y1)//(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

#%%######################### start	######################


for itr in data:
	file_name_json = data[itr]["filename"]
	sub_count = 0							 # Contains count of masks for a single ground truth image
	
	if len(data[itr]["regions"]) > 1:
		for _ in range(len(data[itr]["regions"])):
			key = file_name_json[:-4] + "*" + str(sub_count+1)
			add_to_dict(data, itr, key, sub_count)
			sub_count += 1
	else:
		add_to_dict(data, itr, file_name_json[:-4], 0)

# save the classes JSON file
for folder in [dst_folder, mask_color_folder]:
	class_file= os.path.join(folder, 'classes.json')
	dict_classes = save_classes_json(class_file, list_classes)
# save binary mask
class_filename= os.path.join(mask_folder, 'classes.json')
dict_classes_binary={'Background':0, "road_damage":255}
save_json(class_filename, dict_classes_binary)

# genarate masks
list_images= os.listdir(source_folder)
# flag: input
print(f"\n --> The annotation json file contains {len(file_bbs)} bbs : \n - classes {dict_classes} \n - {len(list_images)} images" )

for file_path in list_images:
	# to_save_folder = os.path.join(images_folder, file_name[:-4])
	curr_img = os.path.join(source_folder, file_path)
	# copy image to new location
	filename=os.path.join(image_folder, os.path.basename(file_path))
	# flag: 
	# os.rename(curr_img,filename)
			
# For each entry in dictionary, generate mask and save in correponding folder
binary_mask=False
image_filename_old=''
label, nb_labels=-1, np.inf
colors ={str(dict_classes[k]) : np.random.randint(10,255,3) for k in list(dict_classes.keys())}
time.sleep(3)
for itr in tqdm(file_bbs):
	classe=file_bbs[itr]['classe']
	num_masks = itr.split("*")
	img_id=num_masks[0] 
	image_filename=[path for path in list_images if img_id in path]
	if image_filename==[]:
		print(f'- warning: image {img_id} not found!')
		continue
	else:
		image_filename=image_filename[0]

	# generate mask and save in correponding folder
	if image_filename_old!=image_filename:
		if nb_labels<=1 or label!=-1:
			mask0=mask.copy()
			if disp>=1:
				mask=cv2.cvtColor(mask.astype(np.uint8),cv2.COLOR_GRAY2RGB) 
				found_classes=','.join([get_label_name(dict_classes, id) for id in np.unique(mask)])
				#color the mask
				color_the_mask(mask, colors)
				masked_img=cv2.addWeighted(img, 1, mask, 0.4, 0.0)

				# draw the object classes/labels
				for label_name,center in labels_names:
					cv2.putText(masked_img, label_name,center ,
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
					cv2.putText(masked_img, f'Image ID: {image_filename_old}', 
										(int(0.01*MASK_WIDTH), int(0.1*MASK_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX,
										0.6, (255,255,255), 2)
				cv2.imshow(f"Annotation",masked_img)
				time.sleep(2)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				
				# verify the mask
				resp=str(input('\n\n -> do you accept the label? (n:no , enter:yes)):'))
				if resp!='n':
					print(f'-> saving the accepted annotation {image_filename_old}')

			else:
				resp='y'
#%% save the masks/Anotation
			if resp!='n':
				# copy the image
				img_filename=os.path.join(image_folder, os.path.basename(filename)) 
				cv2.imwrite(img_filename, img)
				# os.remove(filename) 
				# save the color mask
				cv2.imwrite(os.path.join(mask_color_folder, img_id + ".png") , mask0)
				
				# save the bounding boxes
				nb_box, label_boxes=create_object_boxes(img_filename, mask, dict_classes, annotation_directory=dst_folder, disp=disp)

				# save the W/B mask
				mask0[mask0!=0]=255
				cv2.imwrite(os.path.join(mask_folder, img_id + ".png") , mask0)
				cnt_img += 1
				del mask0
				gc.collect()
				# input(f'\n flag:\n---> {nb_box} bouding boxes saved {label_boxes}')

		# get the new image
		filename=os.path.join(source_folder,image_filename)
		img=cv2.imread(filename)
		MASK_WIDTH, MASK_HEIGHT, channels = img.shape
		# initate new empty mask 
		mask = np.zeros((MASK_WIDTH, MASK_HEIGHT))
		image_filename_old=image_filename
		labels_names=[]
	try:
		arr = np.array(file_bbs[itr]['all_points'])
	except:
		if disp>=1:
			print("Not found:", itr)
		continue
	
	# save the black/white mask 
	label=int(dict_classes[classe])
	nb_labels=len(num_masks) 
	count += 1
	#define the segment center
	update_object_name(arr, labels_names, MASK_WIDTH, MASK_HEIGHT)
	# update the colorfull mask
	cv2.fillPoly(mask, [arr], color=(label))
	if disp>=2:
		print(f'\n\n --> file_bbs [{label}:{classe}]: \n{arr}')

print(f"\n {count} object in {cnt_img} images/masks/bboxes are exported saved in:\n {dst_folder}")