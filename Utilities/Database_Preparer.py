# Cria imagens correspondentes de Edges usando o m√©todo de Canny

# Dataset usado: https://www.kaggle.com/prondeau/the-car-connection-picture-dataset

# Imports
import cv2 as cv
import numpy as np
import os
from os.path import isfile, join
import time

# Par‚metros e constantes
thresh = 100 * 2/3
ratio = 3 # Canny recommended a upper:lower ratio between 2:1 and 3:1.
kernel_size = 3
test_ratio = 0.2
img_size = 256
file_prefix = "car_edges_"
print_interval = 100

read_folder = "60k_car_dataset_selected/"
save_folder = "60k_car_dataset_selected_edges/"
save_folder_train = save_folder + "train/"
save_folder_test = save_folder + "test/"

#%% FUN«’ES

# Detector de bordas de Canny
def CannyEdges(img, threshold):
    img_blur = cv.blur(img, (3,3))
    edges = cv.Canny(img_blur, threshold, threshold*ratio, kernel_size)
    edges = (-1*edges)+255 # inverte preto com branco
    return edges

# Escala a imagem para a menor dimens„o ficar com img_size, mantendo o aspect ratio
def ResizeAspectRatio(img, img_size):
    
    width = img.shape[1]
    height = img.shape[0]
    
    if width < height:
        new_w = img_size
        new_h = int(height * new_w / width)
    else:
        new_h = img_size
        new_w = int(width * new_h / height)
        
    dim = (new_w, new_h)
    resized = cv.resize(img, dim, interpolation = cv.INTER_NEAREST)
    return resized

# Faz um center crop
def CenterCrop(img, crop_size):
    
    width = img.shape[1]
    height = img.shape[0]
    
    cx = int(width/2) 
    cy = int(height/2)
    
    half_crop = int(crop_size/2) 
	
    crop_img = img[cy-half_crop:cy+half_crop, cx-half_crop:cx+half_crop, :]
    return crop_img

def Make3Channel(img, img_size):
    
    img_3c = np.zeros((img_size, img_size, 3), dtype = 'uint8')
    img_3c[:,:,0] = img
    img_3c[:,:,1] = img
    img_3c[:,:,2] = img
    
    return img_3c
    

#%% EXECU«√O

# Encontra os arquivos:
files = [f for f in os.listdir(read_folder) if isfile(join(read_folder, f))]
num_files = len(files)
print("Encontrado {0} arquivos".format(num_files))


# Cria as pastas de saÌda
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if not os.path.exists(save_folder_train):
    os.mkdir(save_folder_train)
    
if not os.path.exists(save_folder_test):
    os.mkdir(save_folder_test)


'''
O arquivo final vai ser um agregado de duas imagens 256x256, lado a lado
A imagem original na esquerda, as bordas de Canny na direita
O prÛprio programa j· faz tambÈm uma separaÁ„o entre train e test
'''

# Para cada arquivo, cria a vers„o canny edges dele
c = 1
t1 = time.time()
for file in files:
    
    if c % print_interval == 0 or c == 1 or c == num_files:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files, 100*c/num_files))    

    # Carrega uma imagem
    img = cv.imread(cv.samples.findFile(read_folder + file))

    # Faz o resize e o center crop
    img = ResizeAspectRatio(img, img_size)
    img = CenterCrop(img, img_size)   

    # Transforma para grayscale para o Canny
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Cria a imagem de edges
    edges = CannyEdges(img_gray, thresh)
    edges = Make3Channel(edges, img_size)

    # Concatena as duas imagens
    img_final = cv.hconcat([img, edges])
    
    # Sorteia se ela ser· imagem de teste ou de treino
    rnum = np.random.rand()
    
    if(rnum < test_ratio):
        dest_folder = save_folder_test
    else:
        dest_folder = save_folder_train
    
    # Prepara o novo filename
    filename = file_prefix + str(c).zfill(len(str(num_files))) + ".jpg"
    cv.imwrite(dest_folder + filename, img_final)
    
    c = c + 1
    
t2 = time.time()
dt = t2-t1

print ('O tempo total foi de {:.2f} min ({:.2f} s)\n'.format(dt/60, dt))