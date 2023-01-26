"""
Perform a segmentation and annotate the results with
bounding boxes and text
"""
import os
import sys
import numpy as np
from glob import glob
import napari

###############################  GUI  ############################
import os
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QGroupBox, QFormLayout, QLabel, QMessageBox,\
                            QLineEdit, QFileDialog, QPushButton, QDialog, QCheckBox, QVBoxLayout
# from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
from PyQt5.QtCore import *

from magicgui import magicgui
from napari.types import ImageData, LabelsData, LayerDataTuple
from magicgui.widgets import FunctionGui

# import utils
from lib.utils import *


class Manual_Annotation_App(QWidget):

    def __init__(self, root_folder, root_annotation, img_ext_list, invert):
        super().__init__()
        self.root_folder=root_folder
        self.annotation_directory=''
        self.root_annotation=root_annotation
        self.img_ext_list = img_ext_list
        self.invert=invert
        # get the data input form attributs.
        self.dialog_widget = main_windows(root_folder=self.root_folder, root_annotation=self.root_annotation, \
                                         img_ext_list=  self.img_ext_list, invert=self.invert )
        # strating functions
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('Manual annotation')
        self.dialog_widget.show()

## DIALOG Windows 
class main_windows(QDialog):
    def __init__(self, root_folder, root_annotation, img_ext_list, invert):
        super(main_windows, self).__init__()
        self.create_gui_objects()
        # variables
        self.invert = invert
        self.img_ext_list = img_ext_list
        self.mask_ext = '.png'
        self.img_ext = '.jpg'
        self.root_folder=root_folder
        self.root_annotation=root_annotation
        self.annotation_directory=''
        self.mask_path = ''
        self.data_input_path=''
        self.list_images_path()
  
    def create_gui_objects(self):
        # the form attributs.
        self.data_source = QCheckBox("folders")
        self.data_source.setChecked(False)
        self.data_source.stateChanged.connect(lambda:self.btnstate(self.data_source))
        self.root_folder_button = QPushButton('Click')# browse file
        self.root_folder_button.clicked.connect(self.get_data_input_path)
        self.dst_folder_button = QPushButton('Click')# browse file
        self.dst_folder_button.clicked.connect(self.get_data_dst_path)
        self.button_Go_inteactive = QPushButton('Go to inteactive inspection')#QDialogButtonBox(QDialogButtonBox.)
        self.button_Go_inteactive.setFont(QFont("Times", 10, QFont.Bold))
        self.button_Go_inteactive.setEnabled(True)
        self.button_Go_inteactive.clicked.connect(self.run_napari_gui)
        # create the form .
        self.createFormGroupBox()
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(self.button_Go_inteactive)
        self.setLayout(mainLayout)
        self.setWindowTitle("Browse image data")
        self.setGeometry(100,100,640,100)
        self.setWindowFlags(
            Qt.WindowCloseButtonHint
        )
    
    def btnstate(self,b):   
        if b.isChecked() == True:
            b.setText('files')
        else:
            b.setText('folders')
            
    def createFormGroupBox(self):
        self.formGroupBox = QGroupBox("")
        layout = QFormLayout()
        layout.addRow(QLabel("Data type:"), self.data_source)
        layout.addRow(QLabel("Input data  :"), self.root_folder_button)
        layout.addRow(QLabel("destination folder:"), self.dst_folder_button)
        self.formGroupBox.setLayout(layout)

    def get_data_path(self):
        # open select folder dialog
        if self.data_source.isChecked() == True:
            data_path = QFileDialog.getOpenFileName(self, 'Select data image ', self.root_folder, "CT image (*.jpg *.png *.tif  *.tiff)")[0]
            self.root_folder = os.path.dirname(data_path)
        else:
            data_path = QFileDialog.getExistingDirectory(self, 'Select images directory ', self.root_folder)
            self.root_folder = data_path
        return data_path

    def check_data_ready(self):
        # disale go inteactive if one of the data is empthy
        if  os.path.exists(self.data_ref_path) and  os.path.exists(self.data_input_path):
            self.button_Go_inteactive.setEnabled(True)
        else:
            self.button_Go_inteactive.setEnabled(False)

    def list_images_path(self):

        if not os.path.isfile(self.root_folder):
            list_image_paths=[]
            for ext in self.img_ext_list:
                list_image_paths = list_image_paths + glob( os.path.join(self.root_folder, '*'+ext) )
            print(f'\n - Root folder:  {self.root_folder}  ')
            print(f'\n - Total images to be annotation :  {len(list_image_paths)} images ')
            self.data_input_path = list_image_paths
        else:
            self.root_folder= os.path.dirname(self.root_folder)
            self.data_input_path = [self.root_folder]
            print(f'\n - The images to be annotation :\n{self.data_input_path[0]}')
        #update the annotation folder
        self.update_annotation_dir()

    def get_data_input_path(self):
        self.data_input_path = self.get_data_path()
        self.root_folder_button.setText(self.data_input_path)
        # prepare images paths from the input root folder/file 
        self.list_images_path()

    def update_annotation_dir(self):
        self.annotation_directory = os.path.join( self.root_annotation, get_folder_tag_from_folder(self.root_folder) )
        self.dst_folder_button.setText(self.annotation_directory)
        self.root_folder_button.setText(self.root_folder)

    def get_data_dst_path(self):
        self.annotation_directory = self.get_data_path()
        self.update_annotation_dir()

    def run_napari_gui(self):
        # preparing the data
        preapre_annotation_folder(self.data_input_path, self.annotation_directory, \
                                  invert=self.invert, img_ext=self.img_ext, mask_ext=self.mask_ext)

        for image_path in self.data_input_path[:3]:
            # run the inspection on Napari GUI
            ret_val, msg_out = self.Napari_GUI(image_path)
            if ret_val !=0:
                self.showdialog_warnning(title='Data error', message=msg_out)

    def run_manual_annotation(self, save_annotation: int=0, annotation_directory: str='./semi-labeling/auto-annotation') -> LayerDataTuple: 
        # save the mask 
        if save_annotation:
            create_new_directory(annotation_directory)
            import tifffile
            layer = self.viewer.layers['mask'] 
            tifffile.imsave('mask.tif', layer.data) 
            print(f'\n\nsave the mask in {annotation_directory} ')

    def Napari_GUI(self, image_path):
        # run napari
        napari.gui_qt()
        # get image/mask destination paths
        dir, filename, file_extension=path_split(image_path)
        img_path, self.mask_path = get_image_mask_paths(self.annotation_directory, filename, img_ext=self.img_ext, mask_ext=self.mask_ext)
        # visualize the image/mask
        self.viewer= napari.Viewer()
        # get (image,mask) pair
        image = load_image(self.mask_path)
        label_image = load_image(img_path)
        self.viewer.add_image(image, name=get_file_tag(img_path))                       # Adds the image to the viewer and give the image layer a name 
        self. viewer.add_labels(label_image, name='mask')
        flood_widget = magicgui(self.run_manual_annotation, save_annotation={'name':'Save_annotat', 'label':'Save annotation:','widget_type':'CheckBox',  'value':0},
                                                    annotation_directory={ 'name':'dst',  'label': 'mask path:', 'value':self.annotation_directory})
        self.viewer.window.add_dock_widget(flood_widget, area='right')
        return 0, ''
    
    def showdialog_information(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setIcon(QMessageBox.Information)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval

    def showdialog_warnning(self, title, message):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setIcon(QMessageBox.Critical)
        msg.setText(message)
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()
        return retval
    
    def showdialog_question(self, title, message):
        reply = QMessageBox.question(self, title, message,
        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        return reply

def set_GUI_style(app):
    # Force the style to be the same on all OSs:
    app.setStyle("Fusion")
    # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.black)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)

def manual_anotation_verification(root_folder, root_annotation, img_ext_list, output_destintion, invert):
    # parser = prepare_parser()
    # args =  parser.parse_args()
    # root_folder = args.root_folder
    # root_annotation = args.dst
    # img_ext_list = args.img_ext_list
    # invert = args.dst


    # run the GUI
    app = QApplication(sys.argv)
    # define the GUI style
    set_GUI_style(app)
    ex = Manual_Annotation_App(root_folder=root_folder, root_annotation=root_annotation, \
                                img_ext_list=img_ext_list, invert=invert)
    sys.exit(app.exec_())

def show_image_and_mask(image_path,mask_path):
  # load the image
  image = load_image(image_path)
  mask = load_image(mask_path)
  # check if mask exists
  dir, filename, file_extension=path_split(image_path)

  # initialise viewer with coins image
  viewer = napari.view_image(image, name='imag:'+filename, rgb=False)
  # add the labels
  label_layer = viewer.add_labels(mask, name=f'mask [{len(np.unique(mask))} segments]')
  print(f'mask segments={np.unique(mask)} ')
  napari.run()

############################  RUN main  #########################
def prepare_parser():
  import numpy as np
  from argparse import ArgumentParser
  parser = ArgumentParser(description='checking anotation')
  parser.add_argument(
      "--root_folder",
      default='./data',
    #   required=True,
      metavar="DIRECTORY",
      help="Directory where CT data are stored.",
      type=str,
  )
  parser.add_argument(
      "--dst",
      default="./annotations",
      metavar="DIRECTORY",
      help="Directory where the annotations  will be stored."
  )

  return parser

if __name__ == '__main__':
    

    # # checking anotation
    # root_folder = '/media/abdo2020/DATA1/data/labeled-dataset/segmentation-dataset/DentalPanoramicXrays/Images'
    # root_annotation='/media/abdo2020/DATA1/Datasets/images-dataset/labeled-data/segmentation-dataset'
    # img_ext_list = ['.jpeg', '.jpg','.png','.tif','.tiff']
    # output_destintion= 'outputs/'
    # invert = False  # invert the color of the save jpg images [please check/verify a sample of saved images      
    # manual_anotation_verification(root_folder, root_annotation, img_ext_list, output_destintion, invert)

    # show the image and mask
    image_path = '/media/abdo2020/DATA1/data/labeled-dataset/segmentation-dataset/HAIS_DATABASE-medium-speed__CSI_CAMERA/images/2022-10-31-17h-34min-50sec__CSI_CAMERA__463.jpg' 
    mask_path = '/media/abdo2020/DATA1/data/labeled-dataset/segmentation-dataset/HAIS_DATABASE-medium-speed__CSI_CAMERA/masks/2022-10-31-17h-34min-50sec__CSI_CAMERA__463.png' 
    show_image_and_mask(image_path,mask_path)


