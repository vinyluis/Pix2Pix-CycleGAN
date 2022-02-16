## Definition of the losses for the GANs used on this project

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

#%% DEFINIÇÃO DAS LOSSES

# As loss de GAN serão binary cross-entropy, pois estamos tentando fazer uma classificação binária (real vs falso)
BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)

## LOSSES 

# Loss adversária do discriminador
def discriminator_loss(disc_real_output, disc_fake_output):
    """Calcula a loss dos discriminadores usando BCE.

    Quando a imagem é real, a saída do discriminador deve ser 1 (ou uma matriz de 1s)
    Quando a imagem é sintética, a saída do discriminador deve ser 0 (ou uma matriz de 0s)
    O BCE (Binary Cross Entropy) avalia o quanto o discriminador acertou ou errou.
    """
    real_loss = BCE(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = BCE(tf.zeros_like(disc_fake_output), disc_fake_output)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss, real_loss, fake_loss

# Loss adversária do gerador
def generator_loss(disc_fake_output):
    """Calcula a loss do gerador usando BCE.

    O gerador quer "enganar" o discriminador, então nesse caso ele é reforçado quando
    a saída do discriminador é 1 (ou uma matriz de 1s) para uma entrada de imagem sintética.
    O BCE (Binary Cross Entropy) avalia o quanto o discriminador acertou ou errou.
    """
    return BCE(tf.ones_like(disc_fake_output), disc_fake_output)

# Loss de consistência de ciclo - CycleGAN
def cycle_loss(real_image, cycled_image):
    """Calcula a loss de consistência de ciclo da rede.

    A loss de consistência de ciclo indica que uma imagem ciclada deve ser muito parecida
    (ou idealmente idêntica) à imagem original. Quanto mais diferentes forem, maior o erro
    acumulado de transformação da rede.
    """
    cycle_loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return cycle_loss

# Identity loss - CycleGAN
def identity_loss(real_image, same_image):
    """Calcula a loss de identidade dos geradores

    Quando uma imagem do domínio A passa por um gerador que transforma uma imagem PARA o 
    domínio A, o gerador não deve fazer grandes mudanças.
    Por isso, a imagem real deve ser igual à mesma imagem quando passa pelo gerador.
    """
    id_loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return id_loss

# Loss completa de gerador
def generator_loss_pix2pix(disc_fake_output, fake_img, target, lambda_l1):
    """Calcula a loss de gerador usando BCE no framework Pix2Pix.

    O gerador quer "enganar" o discriminador, então nesse caso ele é reforçado quando
    a saída do discriminador é 1 (ou uma matriz de 1s) para uma entrada de imagem sintética.
    O BCE (Binary Cross Entropy) avalia o quanto o discriminador acertou ou errou.

    O framework Pix2Pix inclui também a loss L1 (distância absoluta pixel a pixel) entre a
    imagem gerada e a imagem objetivo (target), para direcionar o aprendizado do gerador.
    """
    gan_loss = BCE(tf.ones_like(disc_fake_output), disc_fake_output)
    l1_loss = tf.reduce_mean(tf.abs(target - fake_img))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss


## WASSERSTEIN

'''
Wasserstein GAN - Gradient Penalty (WGAN-GP): A WGAN tem uma forma muito bruta de assegurar a continuidade de Lipschitz, então
os autores criaram o conceito de Gradient Penalty para manter essa condição de uma forma mais suave.
- O gerador tem a MESMA loss da WGAN
- O discriminador, em vez de ter seus pesos limitados pelo clipping, ganha uma penalidade de gradiente que deve ser calculada
'''
def loss_wgangp_generator_unsupervised(disc_generated_output):
    """Calcula a loss de wasserstein com gradient-penalty (WGAN-GP) para o gerador."""
    # O output do discriminador é de tamanho BATCH_SIZE x 1, o valor esperado é a média
    gan_loss = -tf.reduce_mean(disc_generated_output)
    return gan_loss

def loss_wgangp_generator_supervised(disc_generated_output, gen_output, target, lambda_l1):
    """Calcula a loss de wasserstein com gradient-penalty (WGAN-GP) para o gerador."""
    # O output do discriminador é de tamanho BATCH_SIZE x 1, o valor esperado é a média
    gan_loss = -tf.reduce_mean(disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def loss_wgangp_discriminator(disc, disc_real_output, disc_generated_output, real_img, generated_img, lambda_gp):
    """Calcula a loss de wasserstein com gradient-penalty (WGAN-GP) para o discriminador."""
    fake_loss = tf.reduce_mean(disc_generated_output)
    real_loss = tf.reduce_mean(disc_real_output)
    gp = gradient_penalty(disc, real_img, generated_img)
    total_disc_loss = total_disc_loss = -(real_loss - fake_loss) + lambda_gp * gp + (0.001 * tf.reduce_mean(disc_real_output**2))
    return total_disc_loss, real_loss, fake_loss, gp

def loss_wgangp_discriminator_conditional(disc, disc_real_output, disc_generated_output, real_img, generated_img, target, lambda_gp):
    """Calcula a loss de wasserstein com gradient-penalty (WGAN-GP) para o discriminador."""
    fake_loss = tf.reduce_mean(disc_generated_output)
    real_loss = tf.reduce_mean(disc_real_output)
    gp = gradient_penalty_conditional(disc, real_img, generated_img, target)
    total_disc_loss = total_disc_loss = -(real_loss - fake_loss) + lambda_gp * gp + (0.001 * tf.reduce_mean(disc_real_output**2))
    return total_disc_loss, real_loss, fake_loss, gp

@tf.function
def gradient_penalty(discriminator, real_img, fake_img, training):
    """Calcula a penalidade de gradiente para a loss de wassertein-gp (WGAN-GP)."""
    # Get the Batch Size
    batch_size = real_img.shape[0]

    # Calcula gamma
    gamma = tf.random.uniform([batch_size, 1, 1, 1])

    # Calcula a imagem interpolada
    interpolated = real_img * gamma + fake_img * (1 - gamma)

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)

        # 1. Get the discriminator output for this interpolated image.
        if training == 'direct':
            pred = discriminator(interpolated, training=True) # O discriminador usa duas imagens como entrada
        elif training == 'progressive':
            pred = discriminator(interpolated)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def gradient_penalty_conditional(disc, real_img, generated_img, target):
    """Calcula a penalidade de gradiente para a loss de wassertein-gp (WGAN-GP).
    Adaptada para o uso em discriminadores condicionais.
    """
    # Get the Batch Size
    batch_size = real_img.shape[0]

    # Calcula gamma
    gamma = tf.random.uniform([batch_size, 1, 1, 1])

    # Calcula a imagem interpolada
    interpolated = real_img * gamma + generated_img * (1 - gamma)
    
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        
        # 1. Get the discriminator output for this interpolated image.
        pred = disc([interpolated, target], training=True) # O discriminador usa duas imagens como entrada

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated])[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp