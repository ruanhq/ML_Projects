import numpy as np
import tensorflow as tf
import os as os
from dataset import cache
from Train_CD import Model
import cv2,sys
import argparse
from pathlib import Path


#Breaking images:
def break_image(test_image, size):
    ###
    h,w= np.shape(test_image)[0],np.shape(test_image)[1]
    broken_image = []
    #set the numbers of images we want to split.
    h_no = h//size
    w_no = w//size
    h=h_no*size
    w=w_no*size
    #########
    #Extract the information of each sub-block:
    for i in range(0,h_no):
        for j in range(0,w_no):
            split = test_image[size*i:size*(i+1),size*j:size*(j+1),:]
            broken_image.append(split); 
            
    return broken_image,h,w,h_no,w_no


#########
#Process the images from the folder.
class Dataset_test:
    def __init__(self, in_dir, exts='.jpg'):
        # Extend the input directory to the full path.
        in_dir = os.path.abspath(in_dir)

        # Input directory.
        self.in_dir = in_dir
        
        model=Model(in_dir)
        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Filenames for all the files in the test-set
        self.filenames = []

        # Class-number for each file in the test-set.
        self.class_numbers_test = []

        # Total number of classes in the data-set.
        self.num_classes = model.num_classes
        
        # If it is a directory.
        if os.path.isdir(in_dir):
          
            # Get all the valid filenames in the dir
            self.filenames = self._get_filenames_and_paths(in_dir)
         
        else:
            print("Invalid Directory")
        self.images = self.load_images(self.filenames)
        
    def _get_filenames_and_paths(self, dir):
        """
        Create and return a list of filenames with matching extensions in the given directory.
        :param dir:
            Directory to scan for files. Sub-dirs are not scanned.
        :return:
            List of filenames. Only filenames. Does not include the directory.
        """

        # Initialize empty list.
        filenames = []

        # If the directory exists.
        if os.path.exists(dir):
            # Get all the filenames with matching extensions.
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    path = os.path.join(self.in_dir, filename)
                    filenames.append(os.path.abspath(path))

        return filenames

    #Loading the images.
    def load_images(self,image_paths):
        # Load the images from disk.
        images = [cv2.imread(path) for path in image_paths]
    
        # Convert to a numpy array and returns it in the form of [num_images,size,size,channel]
        return np.asarray(images)

#Arguments:
def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing Network')
    parser.add_argument('--in_dir',dest='in_dir',type=str,default='test_images')
    parser.add_argument('--meta_file',dest='meta_file',type=str,default=None)
    parser.add_argument('--CP_dir',dest='chk_point_dir',type=str,default=None)
    parser.add_argument('--save_dir',type=str,default=os.getcwd())
    return parser.parse_args()

def main(args):
    #File names are saved into a cache file
    args=parse_arguments()
    dataset_test = cache(cache_path='my_dataset_cache_test.pkl',
                    fn=Dataset_test, 
                    in_dir=args.in_dir)
    test_images = dataset_test.images

    ######
    #Starting new graphs:
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            #import the model dir
            try:
                file_=Path(args.meta_file)
                abs_path=file_.resolve()
            except FileNotFoundError:
                sys.exit('Meta File Not found')
            else:
                imported_meta = tf.train.import_meta_graph(args.meta_file)
                       
            if os.path.isdir(args.chk_point_dir):
                imported_meta.restore(sess, tf.train.latest_checkpoint(args.chk_point_dir))
            else:
                sys.exit("Check Point Directory does not exist")
            
            x = graph.get_operation_by_name("x").outputs[0]
            #subsize:
            sub_size = 128

            #Get the prediction
            predictions = graph.get_operation_by_name("predictions").outputs[0]
            
            #Take one image each time, pass it through the network and save the results:
            for counter,image in enumerate(test_images):
                broken_image,h,w,h_no,w_no = break_image(image,128)
        
                output_image = np.zeros((h_no * sub_size,w_no *sub_size,3),dtype = np.uint8)
                                            
                feed_dict = {x: broken_image}
                batch_predictions = sess.run(predictions, feed_dict = feed_dict)
            
                matrix_pred = batch_predictions.reshape((h_no,w_no))
                
                #Concentrate after this for post processing

                for i in range(0,h_no):
                    for j in range(0,w_no):
                        a = matrix_pred[i,j]
                        #If we predict it as the crack, it output 1 otherwise it outputs 0.
                        output_image[sub_size*i: sub_size*(i+1),sub_size*j:128*(j+1),:] = a
            
                cropped_image = image[0:h_no*128,0:w_no*128,:]                    
                pred_image = np.multiply(output_image,cropped_image)

                print("Saved {} Image(s)".format(counter+1))
                cv2.imwrite(os.path.join(args.save_dir,'outfile_{}.jpg'.format(counter+1)), pred_image)
                                
if __name__ == '__main__':
    main(sys.argv)
    
    