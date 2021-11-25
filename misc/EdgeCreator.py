# Cria imagens correspondentes de Edges usando o mÃ©todo de Canny

# Dataset usado: https://www.kaggle.com/prondeau/the-car-connection-picture-dataset

# Imports
import cv2 as cv
import os
from os import listdir
from os.path import isfile, join

# Parâmetros e constantes
thresh = 50
ratio = 3 # Canny recommended a upper:lower ratio between 2:1 and 3:1.
kernel_size = 3

read_folder = "60k_car_dataset_selected/"
save_folder = "60k_car_selected_edges/"

if not os.path.exists(save_folder):
    os.mkdir(save_folder)


# Funções
def CannyThreshold(val, img_gray):
    low_threshold = val
    img_blur = cv.blur(img_gray, (3,3))
    edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    edges = (-1*edges)+255 # inverte preto com branco
    return edges


# Encontra os arquivos:
files = [f for f in listdir(read_folder) if isfile(join(read_folder, f))]
num_files = len(files)
print("Encontrado {0} arquivos".format(num_files))


# Para cada arquivo, cria a versão canny edges dele
c = 1
for file in files:

    print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files, 100*c/num_files))    

    img_real = cv.imread(cv.samples.findFile(read_folder + file))
    img_gray = cv.cvtColor(img_real, cv.COLOR_BGR2GRAY)
    edges = CannyThreshold(thresh, img_gray)
    
    cv.imwrite(save_folder + file, edges)
    
    c = c + 1