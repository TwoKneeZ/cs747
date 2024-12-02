import pandas as pd
import numpy as np
from PIL import Image
import time
from tqdm import tqdm

def find_black_image(images_path, df):
    '''
       create a column of images that are not black i.e (np.mean(img)!=0)
       
       Input
           images_path: path of images which is to be analyzed
           label: contain the dataframe with labeled images
           
       Output
           Column indicating the image is black or not
    '''
    list_image = [l for l in df['image']]
    
    #return 1 if image is black and 0 is image is not_black
    return [1 if np.mean(np.array(Image.open(images_path + img).convert('L'))) < 1e-5 else 0 for img in tqdm(list_image)]

if __name__ == "__main__":
    
    #start the timer
    start_time_first = time.time()

    #read csv
    trainLabels = pd.read_csv('trainLabels.csv')
 
    #add .jpeg in image column 
    trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
    
    #initialise black column as nan 
    trainLabels['black'] = np.nan
    
    trainLabels['black'] = find_black_image(images_path = "train_crop_256/", df = trainLabels)
    black_count = (trainLabels['black']==1).sum()
    print(f'total black images:{black_count}')
    
    trainLabels.to_csv('trainLabels_black_images.csv', index=False, header=True)
#     #it will remove all 1 form csv i.e black image
#     '''
#         .loc = gets rows (or columns) with particular labels from the index. 
#         this method will only take images which has value 0(i.e not a black image)
#         in black column and it will ignore the black images that has value 1(i.e black images)
#         in black column
#     '''
#     trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    
#     #save as csv
#     trainLabels.to_csv('E:/Tirth/trainlabel_zip/trainLabels_csv/trainLabels_find_black_images.csv', index=False, header=True)
#     #time elapsed    
#     print("--- %s seconds ---" % (time.time() - start_time_first))    