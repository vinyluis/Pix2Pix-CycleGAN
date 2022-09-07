# Cria imagens correspondentes de Edges usando o método de Canny

# Dataset usado: https://www.kaggle.com/prondeau/the-car-connection-picture-dataset

# Imports
import cv2 as cv
import numpy as np
import os
from os.path import isfile, join
import time

# Parâmetros e constantes
# thresh = 100 * 2/3
thresh = 100 * 1 / 2
ratio = 3  # Canny recommended a upper:lower ratio between 2:1 and 3:1.
kernel_size = 3
test_ratio = 0.2
img_size = 64
print_interval = 100

limit_iterations = False
num_limit_iterations = 5

read_folder = "C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/Anime Faces Dataset/data/"
save_folder = "C:/Users/T-Gamer/OneDrive/Vinicius/01-Estudos/00_Datasets/anime_faces_edges_split/"

save_folder_train = save_folder + "train"
save_folder_test = save_folder + "test"

folder_suffix_A = "A/"
folder_suffix_B = "B/"

file_prefixA = "anime_faces_"
file_prefixB = "anime_edges_"


# %% FUNÇÕES

# Detector de bordas de Canny
def CannyEdges(img, threshold):
    img_blur = cv.blur(img, (3, 3))
    edges = cv.Canny(img_blur, threshold, threshold * ratio, kernel_size)
    edges = (-1 * edges) + 255  # inverte preto com branco
    return edges


def ResizeAspectRatio(img, img_size):

    # Escala a imagem para a menor dimensão ficar com img_size, mantendo o aspect ratio

    width = img.shape[1]
    height = img.shape[0]

    if width < height:
        new_w = img_size
        new_h = int(height * new_w / width)
    else:
        new_h = img_size
        new_w = int(width * new_h / height)

    dim = (new_w, new_h)
    resized = cv.resize(img, dim, interpolation=cv.INTER_NEAREST)
    return resized


def CenterCrop(img, crop_size):

    # Faz um center crop

    width = img.shape[1]
    height = img.shape[0]

    cx = int(width / 2)
    cy = int(height / 2)

    half_crop = int(crop_size / 2)

    crop_img = img[cy - half_crop: cy + half_crop, cx - half_crop: cx + half_crop, :]
    return crop_img


def Make3Channel(img, img_size):

    img_3c = np.zeros((img_size, img_size, 3), dtype='uint8')
    img_3c[:, :, 0] = img
    img_3c[:, :, 1] = img
    img_3c[:, :, 2] = img

    return img_3c


# %% EXECUÇÃO

# Encontra os arquivos:
files = [f for f in os.listdir(read_folder) if isfile(join(read_folder, f))]
num_files = len(files)
print("Encontrado {0} arquivos".format(num_files))


# Cria as pastas de saída
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if not os.path.exists(save_folder_train + folder_suffix_A):
    os.mkdir(save_folder_train + folder_suffix_A)

if not os.path.exists(save_folder_train + folder_suffix_B):
    os.mkdir(save_folder_train + folder_suffix_B)

if not os.path.exists(save_folder_test + folder_suffix_A):
    os.mkdir(save_folder_test + folder_suffix_A)

if not os.path.exists(save_folder_test + folder_suffix_B):
    os.mkdir(save_folder_test + folder_suffix_B)


'''
A pasta final vai ser dividida entre A e B, sendo A a imagem original e B as edges
O próprio programa já faz também uma separação entre train e test
'''

# Para cada arquivo, cria a versão canny edges dele
c = 1
t1 = time.time()
for file in files:

    if c % print_interval == 0 or c == 1 or c == num_files:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files, 100 * c / num_files))

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

    # Sorteia se ela será imagem de teste ou de treino
    rnum = np.random.rand()

    if(rnum < test_ratio):
        dest_folder = save_folder_test
    else:
        dest_folder = save_folder_train

    # Prepara o novo filename e salva
    filenameA = file_prefixA + str(c).zfill(len(str(num_files))) + ".jpg"
    cv.imwrite(dest_folder + folder_suffix_A + filenameA, img)

    filenameB = file_prefixB + str(c).zfill(len(str(num_files))) + ".jpg"
    cv.imwrite(dest_folder + folder_suffix_B + filenameB, edges)

    c = c + 1

    if limit_iterations and (c >= num_limit_iterations):
        break

t2 = time.time()
dt = t2 - t1

print('O tempo total foi de {:.2f} min ({:.2f} s)\n'.format(dt / 60, dt))
