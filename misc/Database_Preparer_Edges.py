# Cria imagens correspondentes de Edges usando o método de Canny

# Dataset usado: https://www.kaggle.com/prondeau/the-car-connection-picture-dataset

# Imports
import cv2 as cv
import numpy as np
import os
from os.path import isfile, join
import time
from tqdm import tqdm

# Parâmetros e constantes
thresh = 100 * 2 / 3
ratio = 3  # Canny recommended a upper:lower ratio between 2:1 and 3:1.
kernel_size = 3
test_ratio = 0.05
val_ratio = 0.15
img_size = 256
file_prefix = "car_edges_"

read_folder = "60k_car_dataset_selected/"
save_folder = "60k_car_dataset_selected_edges/"
save_folder_train = save_folder + "train/"
save_folder_test = save_folder + "test/"
save_folder_val = save_folder + "val/"

# %% FUNÇÕES


def CannyEdges(img, threshold):

    # Detector de bordas de Canny

    img_blur = cv.blur(img, (3, 3))
    edges = cv.Canny(img_blur, threshold, threshold * ratio, kernel_size)
    edges = (-1 * edges) + 255  # inverte preto com branco
    return edges


def ResizeAspectRatio(img, img_size):

    # Escala a imagem para a menor dimens�o ficar com img_size, mantendo o aspect ratio

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

    crop_img = img[cy - half_crop: cy + half_crop,
                   cx - half_crop: cx + half_crop, :]
    return crop_img


def Make3Channel(img, img_size):

    img_3c = np.zeros((img_size, img_size, 3), dtype='uint8')
    img_3c[:, :, 0] = img
    img_3c[:, :, 1] = img
    img_3c[:, :, 2] = img

    return img_3c


# %% PREPARAÇÃO DAS PASTAS

# Cria as pastas de saída
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

if not os.path.exists(save_folder_train):
    os.mkdir(save_folder_train)

if not os.path.exists(save_folder_test):
    os.mkdir(save_folder_test)

if not os.path.exists(save_folder_val):
    os.mkdir(save_folder_val)


# %% EXECUÇÃO

'''
O arquivo final vai ser um agregado de duas imagens 256x256, lado a lado
A imagem original na esquerda, as bordas de Canny na direita
O próprio programa já faz também uma separação entre train e test
'''

if __name__ == "__main__":

    # Encontra os arquivos:
    files = [f for f in os.listdir(read_folder) if isfile(join(read_folder, f))]
    num_files = len(files)
    print(f"Encontrado {num_files} arquivos")

    # Para cada arquivo, cria a versão canny edges dele
    t1 = time.time()
    pbar = tqdm(total=num_files)
    for i, file in enumerate(files):

        c = i + 1
        pbar.update()

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

        # Sorteia se ela será imagem de teste, treino ou validação
        rnum = np.random.rand()

        if(rnum <= test_ratio):
            dest_folder = save_folder_test
        elif (rnum > test_ratio) and (rnum <= (test_ratio + val_ratio)):
            dest_folder = save_folder_val
        else:
            dest_folder = save_folder_train

        # Prepara o novo filename
        filename = file_prefix + str(c).zfill(len(str(num_files))) + ".jpg"
        cv.imwrite(dest_folder + filename, img_final)

    t2 = time.time()
    dt = t2 - t1

    print(f'O tempo total foi de {dt/60:.2f} min ({dt:.2f} s)\n')
