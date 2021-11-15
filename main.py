# Image-to-Image Translation with Conditional Adversarial Networks
# https://www.tensorflow.org/tutorials/generative/cyclegan

### Imports
import os
import time
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import models, utils

#%% Weights & Biases

import wandb
wandb.init(project='pix2pix_cyclegan', entity='vinyluis', mode="disabled")
# wandb.init(project='pix2pix_cyclegan', entity='vinyluis', mode="online")

#%% CONFIG TENSORFLOW

# Evita o erro "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)

# Verifica se a GPU está disponível:
print("---- VERIFICA SE A GPU ESTÁ DISPONÍVEL:")
print(tf.config.list_physical_devices('GPU'))
# Verifica se a GPU está sendo usada na sessão
# print("---- VERIFICA SE A GPU ESTÁ SENDO USADA NA SESSÃO:")
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# print(sess)
print("")

#%% HIPERPARÂMETROS E CONFIGURAÇÕES

config = wandb.config # Salva os hiperparametros no Weights & Biases também

# Root do sistema
# base_root = "../"
base_root = ""

# Parâmetros da imagem de entrada
config.IMG_SIZE = 256
config.OUTPUT_CHANNELS = 3
config.IMG_SHAPE = [config.IMG_SIZE, config.IMG_SIZE, config.OUTPUT_CHANNELS]

# Parâmetros da rede
config.LAMBDA_CYCLEGAN = 10 # Controle da proporção das losses de consistência de ciclo e identidade
config.LAMBDA_PIX2PIX = 100 # Controle da proporção da loss L1 com a loss adversária do gerador
config.FIRST_EPOCH = 1
config.EPOCHS = 10
config.USE_ID_LOSS = True
config.LEARNING_RATE = 1e-4
config.ADAM_BETA_1 = 0.5
config.BUFFER_SIZE = 200
config.BATCH_SIZE = 12
config.USE_CACHE = False

'''
Para não dar overflow de memória:
CycleGAN Unet -> Batch = 12
CycleGAN ResNet -> Batch = 2
Pix2Pix Unet -> Batch = 32
Pix2Pix ResNet -> Batch = 16

Ao usar o dataset de carros, setar USE_CACHE em False
'''

# Parâmetros de checkpoint
config.SAVE_CHECKPOINT = True
config.CHECKPOINT_EPOCHS = 1
config.KEEP_CHECKPOINTS = 2
config.LOAD_CHECKPOINT = True
config.LOAD_SPECIFIC_CHECKPOINT = False
config.LOAD_CKPT_EPOCH = 5

# Configuração de validação
config.VALIDATION = True
config.CYCLE_TEST = True
config.CYCLES = 10

# Outras configurações
PRINT_DOT = 10
PRINT_NEW_LINE = PRINT_DOT * 10
NUM_TEST_PRINTS = 10
QUIET_PLOT = True

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = ""

# Modelo do gerador. Possíveis = 'resnet', 'unet'
config.gen_model = 'unet'

# Tipo de experimento. Possíveis = 'pix2pix', 'cyclegan'
config.net_type = 'cyclegan'

#%% PREPARA AS PASTAS

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'EXP'
if config.exp != "":
  experiment_folder += config.exp
experiment_folder += '_'
experiment_folder += config.net_type
experiment_folder += '_gen_'
experiment_folder += config.gen_model
if config.net_type == 'cyclegan':
  experiment_folder += '_lambda_'
  experiment_folder += str(config.LAMBDA_CYCLEGAN)
experiment_folder += "/"

### Pastas do dataset
dataset_root = 'F:/Vinicius - HD/OneDrive/Vinicius/01-Estudos/00_Datasets/'

# Datasets CycleGAN
cars_folder = dataset_root + '60k_car_dataset_selected_edges_split/'
simpsons_folder = dataset_root + 'simpsons_image_dataset/'
anime_faces_folder = dataset_root + 'anime_faces_edges_split/'
insects_folder = dataset_root + 'flickr_internetarchivebookimages_prepared/'

# Datasets Pix2Pix
cars_unpaired_folder = dataset_root + '60k_car_dataset_selected_edges/'

# --- Dataset escolhido --- 
dataset_folder = simpsons_folder

# Pastas de treino e teste
train_folder = dataset_folder + 'train'
test_folder = dataset_folder + 'test'
folder_suffix_A = "A/"
folder_suffix_B = "B/"

### Pastas dos resultados
result_folder = experiment_folder + 'results-train/'
result_test_folder = experiment_folder + 'results-test/'
model_folder = experiment_folder + 'model/'

### Cria as pastas, se não existirem
if not os.path.exists(experiment_root):
    os.mkdir(experiment_root)

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

if not os.path.exists(result_folder):
    os.mkdir(result_folder)
    
if not os.path.exists(result_test_folder):
    os.mkdir(result_test_folder)
    
if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
### Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


#%% PREPARAÇÃO DOS INPUTS

# Valida os datasets usados
if config.net_type == 'cyclegan':
    if not(dataset_folder == cars_folder or dataset_folder == simpsons_folder or
           dataset_folder == anime_faces_folder or dataset_folder == insects_folder):
        raise utils.DatasetError(config.net_type, dataset_folder)

elif config.net_type == 'pix2pix':
    if not(dataset_folder == cars_unpaired_folder):
        raise utils.DatasetError(config.net_type, dataset_folder)

# Prepara os inputs
if config.net_type == 'cyclegan':
    train_A = tf.data.Dataset.list_files(train_folder + folder_suffix_A + '*.jpg')
    train_A = train_A.map(lambda x: utils.load_image_train_cyclegan(x, config.IMG_SIZE, config.OUTPUT_CHANNELS))
    if config.USE_CACHE:
        train_A = train_A.cache()
    train_A = train_A.shuffle(config.BUFFER_SIZE)
    train_A = train_A.batch(config.BATCH_SIZE)
    
    train_B = tf.data.Dataset.list_files(train_folder + folder_suffix_B + '*.jpg')
    train_B = train_B.map(lambda x: utils.load_image_train_cyclegan(x, config.IMG_SIZE, config.OUTPUT_CHANNELS))
    if config.USE_CACHE:
        train_B = train_B.cache()
    train_B = train_B.shuffle(config.BUFFER_SIZE)
    train_B = train_B.batch(config.BATCH_SIZE)

    test_A = tf.data.Dataset.list_files(test_folder + folder_suffix_A + '*.jpg')
    test_A = test_A.map(lambda x: utils.load_image_test_cyclegan(x, config.IMG_SIZE))
    if config.USE_CACHE:
        test_A = test_A.cache()
    test_A = test_A.shuffle(config.BUFFER_SIZE)
    test_A = test_A.batch(config.BATCH_SIZE)
    
    test_B = tf.data.Dataset.list_files(test_folder + folder_suffix_B + '*.jpg')
    test_B = test_B.map(lambda x: utils.load_image_test_cyclegan(x, config.IMG_SIZE))
    if config.USE_CACHE:
        test_B = test_B.cache()
    test_B = test_B.shuffle(config.BUFFER_SIZE)
    test_B = test_B.batch(config.BATCH_SIZE)

elif config.net_type == 'pix2pix':
    train_dataset = tf.data.Dataset.list_files(train_folder + '/*.jpg')
    train_dataset = train_dataset.map(lambda x: utils.load_image_train_pix2pix(x, config.IMG_SIZE, config.OUTPUT_CHANNELS))
    if config.USE_CACHE:
        train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
    train_dataset = train_dataset.batch(config.BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(test_folder + '/*.jpg')
    test_dataset = test_dataset.map(lambda x: utils.load_image_test_pix2pix(x, config.IMG_SIZE))
    if config.USE_CACHE:
        test_dataset = test_dataset.cache()
    test_dataset = test_dataset.batch(config.BATCH_SIZE)

#%% DEFINIÇÃO DAS LOSSES

# As loss de GAN serão binary cross-entropy, pois estamos tentando fazer uma classificação binária (real vs falso)
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

## LOSSES CYCLEGAN

# Loss adversária do discriminador
def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = loss_obj(tf.zeros_like(disc_fake_output), disc_fake_output)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss

# Loss adversária do gerador
def generator_loss(disc_fake_output):
  return loss_obj(tf.ones_like(disc_fake_output), disc_fake_output)

# Loss de consistência de ciclo - CycleGAN
def cycle_loss(real_image, cycled_image):
  cycle_loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
  return cycle_loss

# Identity loss - CycleGAN
def identity_loss(real_image, same_image):
  id_loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return id_loss

# Loss completa de gerador
def generator_loss_pix2pix(disc_fake_output, fake_img, target):
    gan_loss = loss_obj(tf.ones_like(disc_fake_output), disc_fake_output)
    l1_loss = tf.reduce_mean(tf.abs(target - fake_img))
    total_gen_loss = gan_loss + (config.LAMBDA_PIX2PIX * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

        
#%% FUNÇÕES DE TREINAMENTO

# FUNÇÕES DE TREINAMENTO DA CYCLEGAN

@tf.function
def train_step_cyclegan(gen_g, gen_f, disc_x, disc_y, real_x, real_y, 
                        gen_g_optimizer, gen_f_optimizer, disc_x_optimizer, disc_y_optimizer):

  with tf.GradientTape(persistent=True) as tape:
    # Gera as imagens falsas e as cicladas
    fake_y = gen_g(real_x, training=True)
    cycled_x = gen_f(fake_y, training=True)

    fake_x = gen_f(real_y, training=True)
    cycled_y = gen_g(fake_x, training=True)

    # Discrimina as imagens reais
    disc_real_x = disc_x(real_x, training=True)
    disc_real_y = disc_y(real_y, training=True)

    # Discrimina as imagens falsas
    disc_fake_x = disc_x(fake_x, training=True)
    disc_fake_y = disc_y(fake_y, training=True)

    # Losses adversárias de gerador
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    # Loss de ciclo
    total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)
    
    # Calcula a loss completa de gerador
    total_gen_g_loss = gen_g_loss + config.LAMBDA_CYCLEGAN * total_cycle_loss
    total_gen_f_loss = gen_f_loss + config.LAMBDA_CYCLEGAN * total_cycle_loss

    # Se precisar, adiciona a loss de identidade
    if config.USE_ID_LOSS:
      same_x = gen_f(real_x, training=True)
      same_y = gen_g(real_y, training=True)
      total_gen_g_loss += config.LAMBDA_CYCLEGAN * 0.5 * identity_loss(real_y, same_y)
      total_gen_f_loss += config.LAMBDA_CYCLEGAN * 0.5 * identity_loss(real_x, same_x)

    # Calcula as losses de discriminador
    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
  
  # Calcula os gradientes
  gen_g_gradients = tape.gradient(total_gen_g_loss, gen_g.trainable_variables)
  gen_f_gradients = tape.gradient(total_gen_f_loss, gen_f.trainable_variables)
  
  disc_x_gradients = tape.gradient(disc_x_loss, disc_x.trainable_variables)
  disc_y_gradients = tape.gradient(disc_y_loss, disc_y.trainable_variables)
  
  # Realiza a atualização dos parâmetros através dos gradientes
  gen_g_optimizer.apply_gradients(zip(gen_g_gradients, gen_g.trainable_variables))
  gen_f_optimizer.apply_gradients(zip(gen_f_gradients, gen_f.trainable_variables))
  
  disc_x_optimizer.apply_gradients(zip(disc_x_gradients, disc_x.trainable_variables))
  disc_y_optimizer.apply_gradients(zip(disc_y_gradients, disc_y.trainable_variables))

def fit_cyclegan(FIRST_EPOCH, EPOCHS, gen_g, gen_f, disc_x, disc_y,
                gen_g_optimizer, gen_f_optimizer, disc_x_optimizer, disc_y_optimizer):
  
    print("INICIANDO TREINAMENTO")
    t0 = time.time()
    for epoch in range(FIRST_EPOCH, EPOCHS+1):
        t1 = time.time()
        print("Época: ", epoch)

        for example_A in test_A.take(1):
            filename_A = "A_to_B_epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            fig = utils.generate_save_images(gen_g, example_A, result_folder, filename_A)
            if QUIET_PLOT:
                    plt.close(fig)
        for example_B in test_B.take(1):
            filename_B = "B_to_A_epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            fig = utils.generate_save_images(gen_f, example_B, result_folder, filename_B)
            if QUIET_PLOT:
                plt.close(fig)

        # Train
        n = 0
        for image_x, image_y in tf.data.Dataset.zip((train_A, train_B)):
            train_step_cyclegan(gen_g, gen_f, disc_x, disc_y, image_x, image_y, 
                                gen_g_optimizer, gen_f_optimizer, disc_x_optimizer, disc_y_optimizer)
        
            # Printa pontinhos
            if (n+1) % PRINT_DOT == 0:
                print('.', end='')
                if (n+1) % (PRINT_NEW_LINE) == 0:
                    print()
            n+=1

        if config.SAVE_CHECKPOINT:
            if (epoch) % config.CHECKPOINT_EPOCHS == 0:
                ckpt_manager.save()
                print ('\nSalvando checkpoint da época {}'.format(epoch))

        dt = time.time() - t1
        print ('\nTempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))

    if config.SAVE_CHECKPOINT:
        ckpt_manager.save()

    dt = time.time() - t0
    print ('\nTempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(EPOCHS, dt/60, dt))


# FUNÇÕES DE TREINAMENTO DA PIX2PIX

@tf.function
def train_step_pix2pix(gen, disc, gen_optimizer, disc_optimizer, input_img, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_img = gen(input_img, training=True)
          
        disc_real_output = disc([input_img, target], training=True)
        disc_fake_output = disc([input_img, fake_img], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_pix2pix(disc_fake_output, fake_img, target)
        disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))
    
    return (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss)

def fit_pix2pix(first_epoch, epochs, train_ds, test_ds, gen, disc, gen_optimizer, disc_optimizer):
    
    print("INICIANDO TREINAMENTO")
    t0 = time.time()
    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        print("Época: ", epoch)

        for example_input, example_target in test_ds.take(1):
            filename = "epoch_" + str(epoch).zfill(len(str(EPOCHS))) + ".jpg"
            fig = utils.generate_save_images_pix2pix(gen, example_input, example_target, result_folder, filename)
            if QUIET_PLOT:
                plt.close(fig)
        
        # Train
        for n, (input_image, target) in train_ds.enumerate():
            train_step_pix2pix(gen, disc, gen_optimizer, disc_optimizer, input_image, target)
            
            # Printa pontinhos
            if (n+1) % 100 == 0:
                print('.', end='')
                if (n+1) % (100*100) == 0:
                    print()
        
        # Salva checkpoint
        if config.SAVE_CHECKPOINT:
            if (epoch) % config.CHECKPOINT_EPOCHS == 0:
                ckpt_manager.save()
                print ('\nSalvando checkpoint da época {}'.format(epoch))
        
        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
    
    if config.SAVE_CHECKPOINT:
        ckpt_manager.save()
    
    dt = time.time() - t0
    print ('Tempo usado para {} épocas foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
    

#%% PREPARAÇÃO DOS MODELOS

# ---- GERADORES
if config.net_type == 'cyclegan':
    if config.gen_model == 'unet':
        generator_g = models.Unet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
        generator_f = models.Unet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    elif config.gen_model == 'resnet':
        generator_g = models.ResNet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
        generator_f = models.ResNet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    else:
        raise utils.GeneratorError(config.gen_model)

elif config.net_type == 'pix2pix':
    if config.gen_model == 'unet':
        generator = models.Unet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    elif config.gen_model == 'resnet':
        generator = models.ResNet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    else:
        raise utils.GeneratorError(config.gen_model)

else:
    raise utils.ArchitectureError(config.net_type)


# ---- DISCRIMINADORES
if config.net_type == 'cyclegan':
    discriminator_x = models.Discriminator_CycleGAN(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    discriminator_y = models.Discriminator_CycleGAN(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')

elif config.net_type == 'pix2pix':
    discriminator = models.Discriminator_Pix2Pix(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')

else:
    raise utils.ArchitectureError(config.net_type)


# ---- OTIMIZADORES
if config.net_type == 'cyclegan':
    generator_g_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    generator_f_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    discriminator_x_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    discriminator_y_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)

elif config.net_type == 'pix2pix':
    generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)

else:
    raise utils.ArchitectureError(config.net_type)

#%% CHECKPOINTS

if config.net_type == 'cyclegan':
    ckpt = tf.train.Checkpoint(generator_g = generator_g,
                            generator_f = generator_f,
                            discriminator_x = discriminator_x,
                            discriminator_y = discriminator_y,
                            generator_g_optimizer = generator_g_optimizer,
                            generator_f_optimizer = generator_f_optimizer,
                            discriminator_x_optimizer = discriminator_x_optimizer,
                            discriminator_y_optimizer = discriminator_y_optimizer)

elif config.net_type == 'pix2pix':
    ckpt = tf.train.Checkpoint(generator = generator,
                            discriminator = discriminator,
                            generator_optimizer = generator_optimizer,
                            discriminator_optimizer = discriminator_optimizer)

else:
    raise utils.ArchitectureError(config.net_type)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep = config.KEEP_CHECKPOINTS)

# Se quiser, carrega um checkpoint específico
if config.LOAD_SPECIFIC_CHECKPOINT:
    ckpt.restore(checkpoint_dir + "/ckpt-" + str(config.LOAD_CKPT_EPOCH))
    FIRST_EPOCH = config.LOAD_CKPT_EPOCH
    EPOCHS = config.LOAD_CKPT_EPOCH - 1

# Se for o caso, recupera o checkpoint mais recente
else:
    EPOCHS = config.EPOCHS
    if config.LOAD_CHECKPOINT:   
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint != None:
            print("Carregando checkpoint mais recente...")
            ckpt.restore(latest_checkpoint)
            FIRST_EPOCH = int(latest_checkpoint.split("-")[1]) + 1
        else:
            FIRST_EPOCH = config.FIRST_EPOCH

#%% EXECUÇÃO

if config.net_type == 'cyclegan':
    fit_cyclegan(FIRST_EPOCH, EPOCHS, generator_g, generator_f, discriminator_x, discriminator_y,
                 generator_g_optimizer, generator_f_optimizer, discriminator_x_optimizer, discriminator_y_optimizer)

elif config.net_type == 'pix2pix':
    fit_pix2pix(FIRST_EPOCH, EPOCHS, train_dataset, test_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer)

else:
    raise utils.ArchitectureError(config.net_type)


#%% TESTE

if config.net_type == 'cyclegan':
    print("\nCriando imagens do conjunto de teste...")

    # Run the trained model on the test dataset
    c = 1
    for inp in test_A.take(NUM_TEST_PRINTS):
        filename = "A_to_B_test_" + str(c).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
        fig = utils.generate_save_images(generator_g, inp, result_test_folder, filename)
        if QUIET_PLOT:
            plt.close(fig)
        c = c + 1
        
    # Run the trained model on the test dataset
    c = 1
    for inp in test_B.take(NUM_TEST_PRINTS):
        filename = "B_to_A_test_" + str(c).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
        fig = utils.generate_save_images(generator_f, inp, result_test_folder, filename)
        if QUIET_PLOT:
            plt.close(fig)
        c = c + 1

elif config.net_type == 'pix2pix':
    print("\nCriando imagens do conjunto de teste...")

    # Run the trained model on the test dataset
    c = 1
    for inp in test_dataset.take(NUM_TEST_PRINTS):
        filename = "test_" + str(c).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
        fig = utils.generate_save_images(generator, inp, result_test_folder, filename)
        if QUIET_PLOT:
            plt.close(fig)
        c = c + 1

else:
    raise utils.ArchitectureError(config.net_type)

#%% VALIDAÇÃO

if config.VALIDATION and config.net_type == 'cyclegan' and dataset_folder == cars_folder: 
    
    print("\nINICIANDO VALIDAÇÃO")
    
    validation_read_prefix = base_root
    validation_read_folder_A = validation_read_prefix + 'validation_images_car_cycle/classA/'
    validation_read_folder_B = validation_read_prefix + 'validation_images_car_cycle/classB/'

    validation_save_prefix = experiment_folder + 'validation_results/'    
    validation_save_folder_A = validation_save_prefix + 'AtoB/'
    validation_save_folder_B = validation_save_prefix + 'BtoA/'
    
    if not os.path.exists(validation_save_prefix):
        os.mkdir(validation_save_prefix)
    
    if not os.path.exists(validation_save_folder_A):
        os.mkdir(validation_save_folder_A)
    
    if not os.path.exists(validation_save_folder_B):
        os.mkdir(validation_save_folder_B)
    
    ## VALIDAÇÃO A TO B
    
    # Encontra os arquivos:
    files = [f for f in os.listdir(validation_read_folder_A) if os.path.isfile(os.path.join(validation_read_folder_A, f))]
    val_size = len(files)
    print("Encontrado {0} arquivos".format(val_size))
    
    c = 1
    for file in files:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, val_size, 100*c/val_size))
        filepath = validation_read_folder_A + file
        image = utils.validation_load_A(filepath, config.IMG_SIZE)
        image = np.expand_dims(image, axis=0)
            
        filename = "validation_results_" + str(c).zfill(len(str(val_size))) + ".jpg"
        fig = utils.generate_save_images(generator_g, image, validation_save_folder_A, filename)
        if QUIET_PLOT:
          plt.close(fig)
        c = c + 1
    
    ## VALIDAÇÃO B TO A
    
    # Encontra os arquivos:
    files = [f for f in os.listdir(validation_read_folder_B) if os.path.isfile(os.path.join(validation_read_folder_B, f))]
    val_size = len(files)
    print("Encontrado {0} arquivos".format(val_size))
    
    c = 1
    for file in files:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, val_size, 100*c/val_size))
        filepath = validation_read_folder_B + file
        image = utils.validation_load_B(filepath, config.IMG_SIZE)
        image = np.expand_dims(image, axis=0)
            
        filename = "validation_results_" + str(c).zfill(len(str(val_size))) + ".jpg"
        fig = utils.generate_save_images(generator_f, image, validation_save_folder_B, filename)
        if QUIET_PLOT:
          plt.close(fig)
        c = c + 1
    
if config.VALIDATION and config.net_type == 'pix2pix' and dataset_folder == cars_unpaired_folder:

    print("\nINICIANDO VALIDAÇÃO")

    validation_read_prefix = base_root
    validation_read_folder = validation_read_prefix + 'validation_images_car_sketches/'

    validation_save_prefix = experiment_folder   
    validation_save_folder = validation_save_prefix + 'validation_results/'

    if not os.path.exists(validation_save_folder):
        os.mkdir(validation_save_folder)

    # Encontra os arquivos:
    from os import listdir
    files = [f for f in listdir(validation_read_folder) if os.path.isfile(os.path.join(validation_read_folder, f))]
    val_size = len(files)
    print("Encontrado {0} arquivos".format(val_size))

    c = 1
    for file in files:
        
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, val_size, 100*c/val_size))
        
        filepath = validation_read_folder + file
        image = utils.validation_load_B(filepath)

        image = np.expand_dims(image, axis=0)
            
        filename = "validation_results_" + str(c).zfill(len(str(val_size))) + ".jpg"
        utils.generate_save_images(generator, image, image, validation_save_folder, filename)
        
        c = c + 1

#%% TESTE DE CICLAGEM

if config.CYCLE_TEST and config.net_type == 'cyclegan':
    
    print("\nINICIANDO TESTE DE MÚLTIPLAS CICLAGENS")
    
    cycle_save_folder = experiment_folder + 'cycle_test/'
    
    if not os.path.exists(cycle_save_folder):
        os.mkdir(cycle_save_folder)
    
    # Múltiplos ciclos A to B
    for image in test_A.take(1):
        
        print("\nA -> B")
        filename = "A_to_B_original.jpg"
        #plt.imsave()
        tf.keras.preprocessing.image.save_img(cycle_save_folder + filename, image[0])
        if not QUIET_PLOT:
          plt.figure()
          plt.imshow(image[0] * 0.5 + 0.5)
        
        for c in range(config.CYCLES):
            
            print("Ciclo "+ str(c+1))
            
            filename = "A_to_B_FwdClassB_cycle" + str(c+1).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
            image = generator_g(image)
            if not QUIET_PLOT:
              plt.figure()
              plt.imshow(image[0] * 0.5 + 0.5)
            tf.keras.preprocessing.image.save_img(cycle_save_folder + filename, image[0])
            
            filename = "A_to_B_BkwClassA_cycle" + str(c+1).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
            image = generator_f(image)
            if not QUIET_PLOT:
              plt.figure()
              plt.imshow(image[0] * 0.5 + 0.5)
            tf.keras.preprocessing.image.save_img(cycle_save_folder + filename, image[0])
    
    
    # Múltiplos ciclos B to A
    for image in test_B.take(1):
        
        print("\nB -> A")
        filename = "B_to_A_original.jpg"
        #plt.imsave()
        tf.keras.preprocessing.image.save_img(cycle_save_folder + filename, image[0])
        if not QUIET_PLOT:
          plt.figure()
          plt.imshow(image[0] * 0.5 + 0.5)
        
        for c in range(config.CYCLES):
            
            print("Ciclo "+ str(c+1))
            
            filename = "B_to_A_FwdClassA_cycle" + str(c+1).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
            image = generator_f(image)
            if not QUIET_PLOT:
              plt.figure()
              plt.imshow(image[0] * 0.5 + 0.5)
            tf.keras.preprocessing.image.save_img(cycle_save_folder + filename, image[0]) 
            
            filename = "B_to_A_BkwClassB_cycle" + str(c+1).zfill(len(str(NUM_TEST_PRINTS))) + ".jpg"
            image = generator_g(image)
            if not QUIET_PLOT:
              plt.figure()
              plt.imshow(image[0] * 0.5 + 0.5)
            tf.keras.preprocessing.image.save_img(cycle_save_folder + filename, image[0])