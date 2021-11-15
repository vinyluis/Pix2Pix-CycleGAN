import cv2 as cv
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

#%% FUNÇÕES DE APOIO

# Funções básicas

def normalize(image):
  # normalizing the images to [-1, 1]
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def resize(image, height, width):
    image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image

# Funções CycleGAN

def random_crop_cyclegan(image, img_size, num_channels):
  cropped_image = tf.image.random_crop(value = image, size = [img_size, img_size, num_channels])
  return cropped_image

def random_jitter_cyclegan(image, img_size, num_channels):
  # resizing to 286 x 286 x 3
  new_size = int(img_size * 1.117)
  image = tf.image.resize(image, [new_size, new_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  # randomly cropping to 256 x 256 x 3
  image = random_crop_cyclegan(image, img_size, num_channels)
  # random mirroring
  image = tf.image.random_flip_left_right(image)
  return image

def load_image_train_cyclegan(image_file, img_size, output_channels):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  image = tf.cast(image, tf.float32)
  
  if image.shape[0] != img_size:
      image = tf.image.resize(image, [img_size, img_size])
  
  image = random_jitter_cyclegan(image, img_size, output_channels)
  image = normalize(image)
  return image

def load_image_test_cyclegan(image_file, img_size):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)
  image = tf.cast(image, tf.float32)  
  
  if image.shape[0] != img_size:
      image = tf.image.resize(image, [img_size, img_size])
    
  image = normalize(image)
  return image

# Funções Pix2Pix

def load_pix2pix(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    w = tf.shape(image)[1]
    
    w = w // 2
    real_image = image[:, :w, :]
    input_image = image[:, w:, :]
    
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    
    return input_image, real_image

def random_crop_pix2pix(input_image, real_image, img_size, num_channels):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[2, img_size, img_size, num_channels])
    return cropped_image[0], cropped_image[1]

def random_jitter_pix2pix(input_image, real_image, img_size, num_channels):
    # resizing to 286 x 286 x 3
    new_size = int(img_size * 1.117)
    input_image = resize(input_image, new_size, new_size)
    real_image = resize(real_image, new_size, new_size)
    
    # randomly cropping to 256 x 256 x 3
    input_image, real_image = random_crop_pix2pix(input_image, real_image, img_size, num_channels)
    
    if tf.random.uniform(()) > 0.5:
        # random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    
    return input_image, real_image

def load_image_train_pix2pix(image_file, img_size, output_channels):
    input_image, real_image = load_pix2pix(image_file)    
    input_image, real_image = random_jitter_pix2pix(input_image, real_image, img_size, output_channels)
    input_image = normalize(input_image)
    real_image = normalize(real_image)
    return input_image, real_image

def load_image_test_pix2pix(image_file, img_size):
    input_image, real_image = load_pix2pix(image_file)    
    input_image = resize(input_image, img_size, img_size)
    real_image = resize(real_image, img_size, img_size)
    input_image = normalize(input_image)
    real_image = normalize(real_image)
    return input_image, real_image

# Geração de imagens

def generate_images(generator, test_input):
  prediction = generator(test_input)
    
  f = plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  f.show()

  return f
  
def generate_save_images(generator, test_input, save_destination, filename):
  prediction = generator(test_input)
    
  f = plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  f.show()
  
  f.savefig(save_destination + filename)

  return f

def generate_images_pix2pix(generator, test_input, tar):
    prediction = generator(test_input, training=True)
    f = plt.figure(figsize=(15,15))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    f.show()

    return f
     
def generate_save_images_pix2pix(generator, test_input, tar, save_destination, filename):
    prediction = generator(test_input, training=True)
    f = plt.figure(figsize=(15,15))
    
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    f.show()
    
    f.savefig(save_destination + filename)

    return f
    
# Funções para a validação

def CannyEdges(img, threshold = 200/3, ratio = 3, kernel_size = 3):
    img_blur = cv.blur(img, (3,3))
    edges = cv.Canny(img_blur, threshold, threshold*ratio, kernel_size)
    edges = (-1*edges)+255 # inverte preto com branco
    return edges

def ResizeAspectRatio(img, img_size, fit = 'larger'):
    
    # img = imagem de entrada
    # img_size = tamanho do lado que será redimensionado
    # fit = o lado que terá o tamanho definido é o maior ou o menor. default = larger
    
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
    resized = tf.image.resize(img, dim, method = 'bicubic') 
    
    return resized

def ResizedSquare(img, img_size, background = 'white'):
    
    # Faz o resize mantendo o aspect ratio
    img = ResizeAspectRatio(img, img_size)
    
    # Pega as informações da imagem
    img_shape = tf.shape(img)
    w, h = img_shape[0], img_shape[1]

    # Transforma a imagem num quadrado preenchendo o excesso com a cor branca
    paddings = tf.constant([[int((img_size - w)/2), int((img_size - w)/2)], 
                            [int((img_size - h)/2), int((img_size - h)/2)], 
                            [0, 0]], dtype = 'int32')
    img_padded = tf.pad(img, paddings, mode = 'CONSTANT', constant_values = 1)
    
    # Se houve algum problema de arredondamento, faz um resize
    img_padded = tf.image.resize(img_padded, (img_size, img_size))
    
    return img_padded

def validation_load_A(image_file, img_size):

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)    
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = ResizedSquare(image, img_size) 
    
    return image

def validation_load_B(image_file, img_size):

    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    
    if image.shape[-1] != 1:
        image = tf.image.rgb_to_grayscale(image)
    
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    # image = np.where(image>0.5, tf.ones_like(image), tf.zeros_like(image))
    image = ResizedSquare(image, img_size) 
    image = tf.image.grayscale_to_rgb(image)
    
    return image

#%% TRATAMENTO DE EXCEÇÕES
    
class GeneratorError(Exception):
    def __init__(self, gen_model):
        print("O gerador " + gen_model + " é desconhecido")
    
class ArchitectureError(Exception):
    def __init__(self, net_type):
        print("A rede " + net_type + " não está definida")
        
class DatasetError(Exception):
    def __init__(self, net_type, dataset_folder):
        print("O dataset em " + dataset_folder + " não é compatível com a rede " + net_type)