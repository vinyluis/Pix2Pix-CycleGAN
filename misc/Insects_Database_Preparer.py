# Preprocessa as imagens dos simpsons para padronizar o tamanho e faz diferentes crops
# no dataset montado manualmente com imagens dos simpsons

# Imports
import cv2 as cv
import numpy as np
import os
from os.path import isfile, join
import time

# Parâmetros e constantes
test_ratio = 0.05
val_ratio = 0.15
img_size = 256
file_prefix_A = "insects_wingless_"
file_prefix_B = "insects_winged_"
print_interval = 5

read_folder_prefix = "F:/Vinicius - HD/OneDrive/Vinicius/01-Estudos/00_Datasets/flickr_internetarchivebookimages_unprepared/"
read_folder_A = read_folder_prefix + "Sem Asa/"
read_folder_B = read_folder_prefix + "Com Asa/"

save_folder_prefix = "F:/Vinicius - HD/OneDrive/Vinicius/01-Estudos/00_Datasets/flickr_internetarchivebookimages/"
save_folder_train = save_folder_prefix + "train"
save_folder_test = save_folder_prefix + "test"
save_folder_val = save_folder_prefix + "val"

classA_suffix = "A/"
classB_suffix = "B/"

fill_color = (251, 231, 198)

# Cria as pastas de saída
if not os.path.exists(save_folder_prefix):
    os.mkdir(save_folder_prefix)

if not os.path.exists(save_folder_train + classA_suffix):
    os.mkdir(save_folder_train + classA_suffix)

if not os.path.exists(save_folder_train + classB_suffix):
    os.mkdir(save_folder_train + classB_suffix)

if not os.path.exists(save_folder_test + classA_suffix):
    os.mkdir(save_folder_test + classA_suffix)

if not os.path.exists(save_folder_test + classB_suffix):
    os.mkdir(save_folder_test + classB_suffix)

if not os.path.exists(save_folder_val + classA_suffix):
    os.mkdir(save_folder_val + classA_suffix)

if not os.path.exists(save_folder_val + classB_suffix):
    os.mkdir(save_folder_val + classB_suffix)


# %% FUNÇÕES

# Escala a imagem para a menor dimensão ficar com img_size, mantendo o aspect ratio
def ResizeAspectRatio(img, img_size, fit='larger'):

    """
    img = imagem de entrada
    img_size = tamanho do lado que será redimensionado
    fit = o lado que terá o tamanho definido é o maior ou o menor. default = larger
    """

    # Resize mantendo o aspect ratio
    width = img.shape[0]
    height = img.shape[1]

    # Se for larger, faz com que o lado maior seja igual img_size
    if fit == 'larger':
        if width > height:
            new_w = img_size
            new_h = int(height * new_w / width)
        else:
            new_h = img_size
            new_w = int(width * new_h / height)

    # Se não for larger, faz com que o lado menor seja igual img_size
    else:
        if width < height:
            new_w = img_size
            new_h = int(height * new_w / width)
        else:
            new_h = img_size
            new_w = int(width * new_h / height)

    dim = (new_h, new_w)
    resized = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)

    return resized


# Faz um center crop
def CenterCrop(img, crop_size):

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


def ResizeSquare(img, img_size, background='white'):

    # Faz um quadrado preenchendo o excesso de imagem com pixels brancos

    # Faz o resize mantendo o aspect ratio
    img = ResizeAspectRatio(img, img_size)

    # Pega as informações da imagem
    h, w = img.shape[0], img.shape[1]

    # Cria uma matriz
    if background == "white":
        canvas = np.ones((img_size, img_size, 3), np.uint8) * 255
    elif background == "black":
        canvas = np.zeros((img_size, img_size, 3), np.uint8)
    elif background == "as_pixel":
        pixel1 = (img[0][0]).astype(float)
        pixel2 = (img[-1][0]).astype(float)
        pixel3 = (img[0][-1]).astype(float)
        pixel4 = (img[-1][-1]).astype(float)
        pixel = ((pixel1 + pixel2 + pixel3 + pixel4) / 4).astype(int)
        canvas = np.zeros((img_size, img_size, 3), np.uint8)
        canvas[:] = pixel
    else:
        color = tuple(reversed(background))
        canvas = np.zeros((img_size, img_size, 3), np.uint8)
        canvas[:] = color

    # marca o centro
    center = int(img_size / 2)

    # Posiciona a imagem redimensionada por cima da imagem branca
    canvas[center - int(h / 2): center + int(h / 2),
           center - int(w / 2): center + int(w / 2),
           :] = img[0: int(h / 2) * 2, 0: int(w / 2) * 2, :]

    # Se houve algum problema de arredondamento, faz um resize
    final_image = cv.resize(canvas, (img_size, img_size))

    return final_image


def ThreeCrop(img, img_size):

    img = ResizeAspectRatio(img, img_size, fit='smaller')

    height = img.shape[0]
    width = img.shape[1]

    if width > height:
        img_a = img[:, 0: img_size, :]  # left
        img_b = img[:, width - img_size: width, :]  # right

    else:
        img_a = img[0: img_size, :, :]  # top
        img_b = img[height - img_size: height, :, :]  # bottom

    img_c = CenterCrop(img, img_size)

    # Acerta os tamanhos, só por via das dúvidas
    dim = (img_size, img_size)
    img_a = cv.resize(img_a, dim, interpolation=cv.INTER_CUBIC)
    img_b = cv.resize(img_b, dim, interpolation=cv.INTER_CUBIC)
    img_c = cv.resize(img_c, dim, interpolation=cv.INTER_CUBIC)

    return img_a, img_b, img_c


# %% EXECUÇÃO - Classe A

# Primeiro prepara a Classe A

# Encontra os arquivos:
files_A = [f for f in os.listdir(read_folder_A) if isfile(join(read_folder_A, f))]
num_files_A = len(files_A)
print("Encontrado {0} arquivos da classe A".format(num_files_A))

# Para cada arquivo, cria a versão reduzida em forma de quadrado, center crop, left (or top) crop e right (or bottom) crop
# Assim, para cada arquivo, serão gerados quatro novos arquivos
c = 1
t1 = time.time()
for file in files_A:

    if c % print_interval == 0 or c == 1 or c == num_files_A:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files_A, 100 * c / num_files_A))

    # Sorteia se a imagem será imagem de teste ou de treino
    rnum = np.random.rand()

    if(rnum < test_ratio):
        dest_folder = save_folder_test + classA_suffix
    elif(rnum >= test_ratio and rnum <= (test_ratio + val_ratio)):
        dest_folder = save_folder_val + classA_suffix
    else:
        dest_folder = save_folder_train + classA_suffix

    # Carrega uma imagem
    image_file = read_folder_A + file
    img = cv.imread(cv.samples.findFile(image_file))

    # Faz o resize e o center crop
    # img_a, img_b, img_c = ThreeCrop(img, img_size)

    # Faz a versão reduzida em forma de quadrado
    img_d = ResizeSquare(img, img_size, background="as_pixel")

    # Salva os arquivos
    # filename_a = file_prefix_A + str(c).zfill(len(str(num_files_A))) + "_a" + ".jpg"
    # cv.imwrite(dest_folder + filename_a, img_a)

    # filename_b = file_prefix_A + str(c).zfill(len(str(num_files_A))) + "_b" + ".jpg"
    # cv.imwrite(dest_folder + filename_b, img_b)

    # filename_c = file_prefix_A + str(c).zfill(len(str(num_files_A))) + "_c" + ".jpg"
    # cv.imwrite(dest_folder + filename_c, img_c)

    filename_d = file_prefix_A + str(c).zfill(len(str(num_files_A))) + "_d" + ".jpg"
    cv.imwrite(dest_folder + filename_d, img_d)

    c = c + 1

t2 = time.time()
dt = t2 - t1

print(f'O tempo total foi de {dt / 60:.2f} min ({dt:.2f} s)\n')


# %% EXECUÇÃO - Classe B

# Primeiro prepara a Classe B

# Encontra os arquivos:
files_B = [f for f in os.listdir(read_folder_B) if isfile(join(read_folder_B, f))]
num_files_B = len(files_B)
print("Encontrado {0} arquivos da classe B".format(num_files_B))

# Para cada arquivo, cria a versão reduzida em forma de quadrado, center crop, left (or top) crop e right (or bottom) crop
# Bssim, para cada arquivo, serão gerados quatro novos arquivos
c = 1
t1 = time.time()
for file in files_B:

    if c % print_interval == 0 or c == 1 or c == num_files_B:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files_B, 100 * c / num_files_B))

    # Sorteia se a imagem será imagem de teste ou de treino
    rnum = np.random.rand()

    if(rnum < test_ratio):
        dest_folder = save_folder_test + classB_suffix
    elif(rnum >= test_ratio and rnum <= (test_ratio + val_ratio)):
        dest_folder = save_folder_val + classB_suffix
    else:
        dest_folder = save_folder_train + classB_suffix

    # Carrega uma imagem
    image_file = read_folder_B + file
    img = cv.imread(cv.samples.findFile(image_file))

    # Faz o resize e o center crop
    # img_a, img_b, img_c = ThreeCrop(img, img_size)

    # Faz a versão reduzida em forma de quadrado
    img_d = ResizeSquare(img, img_size, background="as_pixel")

    # Salva os arquivos
    # filename_a = file_prefix_B + str(c).zfill(len(str(num_files_B))) + "_a" + ".jpg"
    # cv.imwrite(dest_folder + filename_a, img_a)

    # filename_b = file_prefix_B + str(c).zfill(len(str(num_files_B))) + "_b" + ".jpg"
    # cv.imwrite(dest_folder + filename_b, img_b)

    # filename_c = file_prefix_B + str(c).zfill(len(str(num_files_B))) + "_c" + ".jpg"
    # cv.imwrite(dest_folder + filename_c, img_c)

    filename_d = file_prefix_B + str(c).zfill(len(str(num_files_B))) + "_d" + ".jpg"
    cv.imwrite(dest_folder + filename_d, img_d)

    c = c + 1

t2 = time.time()
dt = t2 - t1

print(f'O tempo total foi de {dt / 60:.2f} min ({dt:.2f} s)\n')
