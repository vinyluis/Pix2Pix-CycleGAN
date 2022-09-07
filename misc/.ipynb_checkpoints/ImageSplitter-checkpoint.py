
# Corta as imagens do dataset de carro selected edges e separa a metade direita da metade esquerda 

# Imports
import cv2 as cv
import os
from os.path import isfile, join
import time

# Parâmetros e constantes
file_prefix_A = "car_"
file_prefix_B = "edges_"
print_interval = 100

read_folder = "../Pix2Pix/60k_car_dataset_selected_edges/"
read_folder_train = read_folder + "train/"
read_folder_test = read_folder + "test/"

save_folder = "60k_car_dataset_selected_edges_split/"
save_folder_train = save_folder + "train"
save_folder_test = save_folder + "test"

save_folder_train_A = save_folder_train + "A/"
save_folder_train_B = save_folder_train + "B/"

save_folder_test_A = save_folder_test + "A/"
save_folder_test_B = save_folder_test + "B/"

#%% FUNÇÕES

# Faz um center crop
def splitter(img):
    width = img.shape[1]    
    half_crop = int(width/2) 
	
    img_A = img[:, 0 : half_crop, :]
    img_B = img[:, half_crop + 1 : width, :]
    
    return img_A, img_B


#%% EXECUÇÃO

# Encontra os arquivos:
files_train = [f for f in os.listdir(read_folder_train) if isfile(join(read_folder_train, f))]
files_test = [f for f in os.listdir(read_folder_test) if isfile(join(read_folder_test, f))]
num_files_train = len(files_train)
num_files_test = len(files_test)
num_files = num_files_train + num_files_test
print("Encontrado {0} arquivos".format(num_files))

# Cria as pastas de saída
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
    
if not os.path.exists(save_folder_train_A):
    os.mkdir(save_folder_train_A)
    
if not os.path.exists(save_folder_train_B):
    os.mkdir(save_folder_train_B)
    
if not os.path.exists(save_folder_test_A):
    os.mkdir(save_folder_test_A)
    
if not os.path.exists(save_folder_test_B):
    os.mkdir(save_folder_test_B)

t1 = time.time()

# Para cada arquivo de treino, separa em A (carro) e B (edges) 
c = 1
for file in files_train:
    
    if c % print_interval == 0 or c == 1 or c == num_files_train:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files_train, 100*c/num_files_train))    

    # Carrega uma imagem
    img = cv.imread(cv.samples.findFile(read_folder_train + file))

    # Separa a metade esquerda da metade direita
    img_A, img_B = splitter(img)
    
    # Prepara o novo filename
    filename_A = "train_" + file_prefix_A + str(c).zfill(len(str(num_files_train))) + ".jpg"
    filename_B = "train_" + file_prefix_B + str(c).zfill(len(str(num_files_train))) + ".jpg"
    cv.imwrite(save_folder_train_A + filename_A, img_A)
    cv.imwrite(save_folder_train_B + filename_B, img_B)
    
    c = c + 1
   
# Para cada arquivo de teste, separa em A (carro) e B (edges) 
c = 1
for file in files_test:
    
    if c % print_interval == 0 or c == 1 or c == num_files_test:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files_test, 100*c/num_files_test))    

    # Carrega uma imagem
    img = cv.imread(cv.samples.findFile(read_folder_test + file))

    # Separa a metade esquerda da metade direita
    img_A, img_B = splitter(img)
    
    # Prepara o novo filename
    filename_A = "test_" + file_prefix_A + str(c).zfill(len(str(num_files_test))) + ".jpg"
    filename_B = "test_" + file_prefix_B + str(c).zfill(len(str(num_files_test))) + ".jpg"
    cv.imwrite(save_folder_test_A + filename_A, img_A)
    cv.imwrite(save_folder_test_B + filename_B, img_B)
    
    c = c + 1
    
    
t2 = time.time()
dt = t2-t1

print ('O tempo total foi de {:.2f} min ({:.2f} s)\n'.format(dt/60, dt))