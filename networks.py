# Modelos usados na CycleGAN e Pix2Pix
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf
import tensorflow_addons as tfa

'''
U-Net e discriminadores baseados na versão do tensorflow_examples
https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py

Geradores CycleGAN e Pix2Pix adaptados do github do paper original
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

'''

#%% BLOCOS

## Básicos
def downsample(filters, size, norm_type = 'instancenorm', apply_norm = True):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    
    if apply_norm:
        if norm_type == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type == 'instancenorm':
            result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU(0.2))
    
    return result

def upsample(filters, size, norm_type = 'instancenorm', apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    if norm_type == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type == 'instancenorm':
        result.add(tfa.layers.InstanceNormalization())
    
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    
    result.add(tf.keras.layers.ReLU())
    
    return result

## Residuais

def residual_block(input_tensor, filters, norm_type = 'instancenorm'):
    
    ''' 
    Cria um bloco resnet baseado na Resnet34
    https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    '''

    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization
    
    x = input_tensor
    skip = input_tensor
    
    # Primeira convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Segunda convolução (kernel = 3, 3)
    x = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3, 3), padding = 'same')(x)
    x = norm_layer()(x)
    
    # Concatenação
    x = tf.keras.layers.Add()([x, skip])
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    return x


#%% GERADORES

# Gerador Pix2Pix
def Pix2Pix_Generator(IMG_SIZE, OUTPUT_CHANNELS):

    # Define os inputs
    inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])

    # Encoder
    x = inputs
    x = downsample(64, 4, norm_type = 'batchnorm', apply_norm = False)(x)
    x = downsample(128, 4, norm_type = 'batchnorm')(x)
    x = downsample(256, 4, norm_type = 'batchnorm')(x)
    x = downsample(512, 4, norm_type = 'batchnorm')(x)
    x = downsample(512, 4, norm_type = 'batchnorm')(x)
    x = downsample(512, 4, norm_type = 'batchnorm')(x)
    x = downsample(512, 4, norm_type = 'batchnorm')(x)
    x = downsample(512, 4, norm_type = 'batchnorm')(x)

    # Decoder
    x = upsample(512, 4, apply_dropout=True, norm_type = 'batchnorm')(x)
    x = upsample(512, 4, apply_dropout=True, norm_type = 'batchnorm')(x)
    x = upsample(512, 4, apply_dropout=True, norm_type = 'batchnorm')(x)
    x = upsample(512, 4, norm_type = 'batchnorm')(x)
    x = upsample(256, 4, norm_type = 'batchnorm')(x)
    x = upsample(128, 4, norm_type = 'batchnorm')(x)
    x = upsample(64, 4, norm_type = 'batchnorm')(x)

    initializer = tf.random_normal_initializer(0., 0.02)
    x = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# Gerador U-Net
def Unet_Generator(IMG_SIZE, OUTPUT_CHANNELS, norm_type = 'instancenorm'):
  inputs = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])

  down_stack = [
    downsample(64, 4, norm_type,  apply_norm = False), # (bs, 128, 128, 64)
    downsample(128, 4, norm_type), # (bs, 64, 64, 128)
    downsample(256, 4, norm_type), # (bs, 32, 32, 256)
    downsample(512, 4, norm_type), # (bs, 16, 16, 512)
    downsample(512, 4, norm_type), # (bs, 8, 8, 512)
    downsample(512, 4, norm_type), # (bs, 4, 4, 512)
    downsample(512, 4, norm_type), # (bs, 2, 2, 512)
    downsample(512, 4, norm_type), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 256)
    upsample(64, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

# Gerador CycleGAN
def CycleGAN_Generator(IMG_SIZE, OUTPUT_CHANNELS, norm_type = 'instancenorm', num_residual_blocks = 9):
    
    '''
    Versão original do gerador utilizado no paper CycleGAN
    '''

    if norm_type == 'batchnorm':
        norm_layer = tf.keras.layers.BatchNormalization
    elif norm_type == 'instancenorm':
        norm_layer = tfa.layers.InstanceNormalization

    initializer = tf.random_normal_initializer(0., 0.02)
    
    # Inicializa a rede
    inputs = tf.keras.layers.Input(shape = [IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS])
    x = inputs
    
    # Primeiras camadas (pré blocos residuais)
    x = tf.keras.layers.ZeroPadding2D([[3, 3],[3, 3]])(x)
    x = tf.keras.layers.Conv2D(filters = 64, kernel_size = (7, 7) , strides = (1, 1), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    #--
    x = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    #--
    x = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3) , strides = (2, 2), padding = "valid", kernel_initializer=initializer, use_bias = True)(x)
    x = norm_layer()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    
    # Blocos Resnet
    for i in range(num_residual_blocks):
        x = residual_block(x, 256)
        
    # print(x.shape)
        
    # Reconstrução da imagem
    x = tf.keras.layers.Conv2DTranspose(filters = 128, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = tf.keras.layers.Conv2DTranspose(filters = 64, kernel_size = (3, 3) , strides = (2, 2), padding = "same", kernel_initializer=initializer, use_bias = True)(x)    
    x = norm_layer()(x)
    x = tf.keras.layers.ReLU()(x)

    # Camadas finais
    x = tf.keras.layers.ZeroPadding2D([[2, 2],[2, 2]])(x)
    x = tf.keras.layers.Conv2D(filters = 3, kernel_size = (7, 7) , strides = (1, 1), padding = "same", kernel_initializer=initializer, use_bias = True)(x)
    x = tf.keras.layers.Activation('tanh')(x)
    
    # print(x.shape)

    # Cria o modelo
    return tf.keras.Model(inputs = inputs, outputs = x)

#%% DISCRIMINADORES

def CycleGAN_Discriminator(IMG_SIZE, OUTPUT_CHANNELS, norm_type ='instancenorm'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='input_image')
    x = inp

    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    
    if norm_type == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type == 'instancenorm':
        norm1 = tfa.layers.InstanceNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    
    return tf.keras.Model(inputs=inp, outputs=last)

def Pix2Pix_Discriminator(IMG_SIZE, OUTPUT_CHANNELS, norm_type ='instancenorm'):
    initializer = tf.random_normal_initializer(0., 0.02)
    
    inp = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[IMG_SIZE, IMG_SIZE, OUTPUT_CHANNELS], name='target_image')
    
    x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
    
    down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
    down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
    down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)
    
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) # (bs, 31, 31, 512)
    
    if norm_type == 'batchnorm':
        norm1 = tf.keras.layers.BatchNormalization()(conv)
    elif norm_type == 'instancenorm':
        norm1 = tfa.layers.InstanceNormalization()(conv)
    
    leaky_relu = tf.keras.layers.LeakyReLU()(norm1)
    
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
    
    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)
    
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


#%% TESTA

#  Só roda quando este arquivo for chamado como main
if __name__ == "__main__":

    # Testa os shapes dos modelos
    IMG_SIZE = 256
    print(f"\n---- IMG_SIZE = {IMG_SIZE}")
    print("Geradores:")
    print("Pix2Pix  ", Pix2Pix_Generator(IMG_SIZE, 3).output.shape)
    print("U-Net    ", Unet_Generator(IMG_SIZE, 3).output.shape)
    print("CycleGAN ", CycleGAN_Generator(IMG_SIZE, 3).output.shape)
    print("Discriminadores:")
    print("CycleGAN ", CycleGAN_Discriminator(IMG_SIZE, 3).output.shape)
    print("Pix2Pix  ", Pix2Pix_Discriminator(IMG_SIZE, 3).output.shape)