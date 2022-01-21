## Main code for Pix2Pix and CycleGAN
## Based on: https://www.tensorflow.org/tutorials/generative/cyclegan
## Created for the Master's degree dissertation
## Vinícius Trevisan 2020 - 2022

### Imports
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from math import ceil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf

# Módulos próprios
import networks, utils, metrics

#%% Weights & Biases
import wandb

# Define o projeto. Como são abordagens diferentes, dois projetos diferentes foram criados no wandb
# Possíveis: 'cyclegan' e 'pix2pix'
wandb_project = 'cyclegan'

# Valida o projeto
if not(wandb_project == 'cyclegan' or wandb_project == 'pix2pix'):
    raise utils.ProjectError(wandb_project)

# wandb.init(project=wandb_project, entity='vinyluis', mode="disabled")
wandb.init(project=wandb_project, entity='vinyluis', mode="online")

#%% CONFIG TENSORFLOW

# Verifica se a GPU está disponível:
print("---- VERIFICA SE A GPU ESTÁ DISPONÍVEL:")
print(tf.config.list_physical_devices('GPU'))
print("")

#%% HIPERPARÂMETROS E CONFIGURAÇÕES

config = wandb.config # Salva os hiperparametros no Weights & Biases também

# Root do sistema
base_root = ""

# Parâmetros da imagem de entrada
config.IMG_SIZE = 256
config.OUTPUT_CHANNELS = 3
config.IMG_SHAPE = [config.IMG_SIZE, config.IMG_SIZE, config.OUTPUT_CHANNELS]

# Parâmetros da rede
config.TRAIN = True
config.LAMBDA_CYCLEGAN = 10 # Controle da proporção das losses de consistência de ciclo e identidade
config.LAMBDA_PIX2PIX = 100 # Controle da proporção da loss L1 com a loss adversária do gerador
config.FIRST_EPOCH = 1
config.EPOCHS = 5
config.USE_ID_LOSS = True
config.LEARNING_RATE = 1e-4
config.ADAM_BETA_1 = 0.5
config.NUM_RESIDUAL_BLOCKS = 6 # Número de blocos residuais do gerador CycleGAN
# BUFFER_SIZE, BATCH_SIZE e USE_CACHE serão definidos em código

# Parâmetros das métricas
config.EVALUATE_IS = True
config.EVALUATE_FID = True
config.EVALUATE_L1 = True
config.EVALUATE_PERCENT_OF_DATASET_TRAIN = 0.10
config.EVALUATE_PERCENT_OF_DATASET_VAL = 0.20
config.EVALUATE_PERCENT_OF_DATASET_TEST = 1.00
config.EVALUATE_TRAIN_IMGS = False # Define se vai usar imagens de treino na avaliação
config.EVALUATE_EVERY_EPOCH = True # Define se vai avaliar em cada época ou apenas no final
# METRIC_SAMPLE_SIZE e METRIC_BATCH_SIZE serão definidas em código, para treino e teste

# Parâmetros de checkpoint
config.SAVE_CHECKPOINT = True
config.CHECKPOINT_EPOCHS = 1
config.KEEP_CHECKPOINTS = 1
config.LOAD_CHECKPOINT = True
config.LOAD_SPECIFIC_CHECKPOINT = False
config.LOAD_CKPT_EPOCH = 5
config.SAVE_MODELS = True

# Configurações de validação
config.VALIDATION = True # Gera imagens da validação
config.EVAL_ITERATIONS = 10 # A cada quantas iterações se faz a avaliação das métricas nas imagens de validação
config.NUM_VAL_PRINTS = 10 # Controla quantas imagens de validação serão feitas. Com -1 plota todo o dataset de validação

# Configurações de teste
config.TEST = True # Teste do modelo
config.NUM_TEST_PRINTS = 500 # Controla quantas imagens de teste serão feitas. Com -1 plota todo o dataset de teste

# Configuração do teste de ciclo e generalização
config.GENERALIZATION = True # Teste de generalização
config.CYCLE_TEST = True # Realiza o teste de ciclo
config.CYCLES = 10 # Quantos ciclos para cada teste de ciclo
config.CYCLE_TEST_PICTURES = 10 # Em quantas imagens será feito o teste de ciclo

# Outras configurações
QUIET_PLOT = True # Controla se as imagens aparecerão na tela, o que impede a execução do código a depender da IDE
SHUTDOWN_AFTER_FINISH = False # Controla se o PC será desligado quando o código terminar corretamente

#%% CONTROLE DA ARQUITETURA

# Código do experimento (se não houver, deixar "")
config.exp = "C01E"

# Modelo do gerador. Possíveis = 'pix2pix', 'unet', 'cyclegan'
config.gen_model = 'cyclegan'

# Tipo de experimento. Possíveis = 'pix2pix', 'cyclegan'
config.net_type = 'cyclegan'

# Valida se o experimento é coerente com o projeto wandb selecionado
if not((wandb_project == 'cyclegan' and config.net_type == 'cyclegan') 
   or (wandb_project == 'pix2pix' and config.net_type == 'pix2pix')):
    raise utils.ProjectMismatch(wandb_project, config.net_type)

# Valida se o número de blocos residuais é válido para o gerador CycleGAN
if config.gen_model == 'cyclegan':
    if not (config.NUM_RESIDUAL_BLOCKS == 6 or config.NUM_RESIDUAL_BLOCKS == 9):
        raise BaseException("O número de blocos residuais do gerador CycleGAN não está correto. Opções = 6 ou 9.") 

#%% PREPARA AS PASTAS

### Prepara o nome da pasta que vai salvar o resultado dos experimentos
experiment_root = base_root + 'Experimentos/'
experiment_folder = experiment_root + 'Exp_'
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

### Pastas dos resultados
result_folder = experiment_folder + 'results-train/'
result_test_folder = experiment_folder + 'results-test/'
result_val_folder = experiment_folder + 'results-val/'
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
    
if not os.path.exists(result_val_folder):
    os.mkdir(result_val_folder)

if not os.path.exists(model_folder):
    os.mkdir(model_folder)
    
### Pasta do checkpoint
checkpoint_dir = experiment_folder + 'checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

#%% DATASET

### Pastas do dataset
dataset_root = '../../0_Datasets/celeba_hq/'

# Datasets CycleGAN
cars_folder = dataset_root + '60k_car_dataset_selected_edges_split/'
simpsons_folder = dataset_root + 'simpsons_image_dataset/'
anime_faces_folder = dataset_root + 'anime_faces_edges_split/'
insects_folder = dataset_root + 'flickr_internetarchivebookimages_prepared/'

# Datasets Pix2Pix
cars_paired_folder = dataset_root + '60k_car_dataset_selected_edges/'
cars_paired_folder_complete = dataset_root + '60k_car_dataset_edges/'

#############################
# --- Dataset escolhido --- #
#############################
dataset_folder = cars_folder

# Pastas de treino e teste
train_folder = dataset_folder + 'train'
val_folder = dataset_folder + 'val'
test_folder = dataset_folder + 'test'
folder_suffix_A = "A/"
folder_suffix_B = "B/"

#%% DEFINIÇÃO DE BATCH SIZE E CACHE

'''
Para não dar overflow de memória é necessário ser cuidadoso com o Batch Size
Ao usar os datasets de carros, setar USE_CACHE em False
'''

# Definiçãodo BUFFER_SIZE
config.BUFFER_SIZE = 100

# Definição do BATCH_SIZE
if config.net_type == 'cyclegan':
    if config.gen_model == 'unet' or config.gen_model == 'pix2pix':
        config.BATCH_SIZE = 4
    elif config.gen_model == 'cyclegan':
        config.BATCH_SIZE = 2
    else:
        config.BATCH_SIZE = 1

elif config.net_type == 'pix2pix':
    if config.gen_model == 'unet' or config.gen_model == 'pix2pix':
        config.BATCH_SIZE = 12
    elif config.gen_model == 'cyclegan':
        config.BATCH_SIZE = 6
    else:
        config.BATCH_SIZE = 10

else:
    config.BATCH_SIZE = 10

# Definição do USE_CACHE
if dataset_folder == cars_folder or dataset_folder == cars_paired_folder or dataset_folder == cars_paired_folder_complete:
    config.USE_CACHE = False

else:
    config.USE_CACHE = True


#%% PREPARAÇÃO DOS INPUTS

print("Carregando o dataset...")

# Valida os datasets usados
if config.net_type == 'cyclegan':
    if not(dataset_folder == cars_folder or dataset_folder == simpsons_folder or
           dataset_folder == anime_faces_folder or dataset_folder == insects_folder):
        raise utils.DatasetError(config.net_type, dataset_folder)

elif config.net_type == 'pix2pix':
    if not(dataset_folder == cars_paired_folder or dataset_folder == cars_paired_folder_complete):
        raise utils.DatasetError(config.net_type, dataset_folder)

# Prepara os inputs
if config.net_type == 'cyclegan':
    train_A = tf.data.Dataset.list_files(train_folder + folder_suffix_A + '*.jpg')
    config.TRAIN_SIZE_A = len(list(train_A))
    train_A = train_A.map(lambda x: utils.load_image_train_cyclegan(x, config.IMG_SIZE, config.OUTPUT_CHANNELS))
    if config.USE_CACHE:
        train_A = train_A.cache()
    train_A = train_A.shuffle(config.BUFFER_SIZE)
    train_A = train_A.batch(config.BATCH_SIZE)
    
    train_B = tf.data.Dataset.list_files(train_folder + folder_suffix_B + '*.jpg')
    config.TRAIN_SIZE_B = len(list(train_B))
    train_B = train_B.map(lambda x: utils.load_image_train_cyclegan(x, config.IMG_SIZE, config.OUTPUT_CHANNELS))
    if config.USE_CACHE:
        train_B = train_B.cache()
    train_B = train_B.shuffle(config.BUFFER_SIZE)
    train_B = train_B.batch(config.BATCH_SIZE)

    test_A = tf.data.Dataset.list_files(test_folder + folder_suffix_A + '*.jpg', shuffle = False)
    config.TEST_SIZE_A = len(list(test_A))
    test_A = test_A.map(lambda x: utils.load_image_test_cyclegan(x, config.IMG_SIZE))
    if config.USE_CACHE:
        test_A = test_A.cache()
    test_A = test_A.batch(1)
    
    test_B = tf.data.Dataset.list_files(test_folder + folder_suffix_B + '*.jpg', shuffle = False)
    config.TEST_SIZE_B = len(list(test_B))
    test_B = test_B.map(lambda x: utils.load_image_test_cyclegan(x, config.IMG_SIZE))
    if config.USE_CACHE:
        test_B = test_B.cache()
    test_B = test_B.batch(1)

    val_A = tf.data.Dataset.list_files(val_folder + folder_suffix_A + '*.jpg', shuffle = False)
    config.VAL_SIZE_A = len(list(val_A))
    val_A = val_A.map(lambda x: utils.load_image_test_cyclegan(x, config.IMG_SIZE))
    if config.USE_CACHE:
        val_A = val_A.cache()
    val_A = val_A.batch(1)

    val_B = tf.data.Dataset.list_files(val_folder + folder_suffix_B + '*.jpg', shuffle = False)
    config.VAL_SIZE_B = len(list(val_B))
    val_B = val_B.map(lambda x: utils.load_image_test_cyclegan(x, config.IMG_SIZE))
    if config.USE_CACHE:
        val_B = val_B.cache()
    val_B = val_B.batch(1)


    print("O dataset de treino A tem {} imagens".format(config.TRAIN_SIZE_A))
    print("O dataset de treino B tem {} imagens".format(config.TRAIN_SIZE_B))
    print("O dataset de teste A tem {} imagens".format(config.TEST_SIZE_A))
    print("O dataset de teste B tem {} imagens".format(config.TEST_SIZE_B))
    print("O dataset de validação A tem {} imagens".format(config.VAL_SIZE_A))
    print("O dataset de validação B tem {} imagens".format(config.VAL_SIZE_B))
    print("")

elif config.net_type == 'pix2pix':
    train_dataset = tf.data.Dataset.list_files(train_folder + '/*.jpg')
    config.TRAIN_SIZE = len(list(train_dataset))
    train_dataset = train_dataset.map(lambda x: utils.load_image_train_pix2pix(x, config.IMG_SIZE, config.OUTPUT_CHANNELS))
    if config.USE_CACHE:
        train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(config.BUFFER_SIZE)
    train_dataset = train_dataset.batch(config.BATCH_SIZE)

    test_dataset = tf.data.Dataset.list_files(test_folder + '/*.jpg', shuffle = False)
    config.TEST_SIZE = len(list(test_dataset))
    test_dataset = test_dataset.map(lambda x: utils.load_image_test_pix2pix(x, config.IMG_SIZE))
    if config.USE_CACHE:
        test_dataset = test_dataset.cache()
    test_dataset = test_dataset.batch(1)

    val_dataset = tf.data.Dataset.list_files(val_folder + '/*.jpg', shuffle = False)
    config.VAL_SIZE = len(list(val_dataset))
    val_dataset = val_dataset.map(lambda x: utils.load_image_test_pix2pix(x, config.IMG_SIZE))
    if config.USE_CACHE:
        val_dataset = val_dataset.cache()
    val_dataset = val_dataset.batch(1)

    print("O dataset de treino tem {} imagens".format(config.TRAIN_SIZE))
    print("O dataset de teste tem {} imagens".format(config.TEST_SIZE))
    print("O dataset de validação tem {} imagens".format(config.VAL_SIZE))
    print("")

#%% DEFINIÇÃO DAS LOSSES

# As loss de GAN serão binary cross-entropy, pois estamos tentando fazer uma classificação binária (real vs falso)
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

## LOSSES 

# Loss adversária do discriminador
def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = loss_obj(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = loss_obj(tf.zeros_like(disc_fake_output), disc_fake_output)
    total_disc_loss = real_loss + fake_loss
    return total_disc_loss, real_loss, fake_loss

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


## MÉTRICAS DE QUALIDADE

'''
Serão avaliadas IS, FID, L1 e ACC de acordo com as flags no início do programa
METRIC_SAMPLE_SIZE e METRIC_BATCH_SIZE serão definidas aqui de acordo com o tamanho
do dataset e o valor em EVALUATE_PERCENT_OF_DATASET
'''

# Configuração dos batches sizes
if dataset_folder == simpsons_folder:
    config.METRIC_BATCH_SIZE = 5 # Não há imagens o suficiente para fazer um batch size muito grande

elif dataset_folder == cars_folder or dataset_folder == cars_paired_folder or dataset_folder == cars_paired_folder_complete:
    config.METRIC_BATCH_SIZE = 36

else:
    config.METRIC_BATCH_SIZE = 10

# Configuração dos sample sizes
if config.net_type == 'cyclegan':
    min_train_ds_size = min(config.TRAIN_SIZE_A, config.TRAIN_SIZE_B)
    min_test_ds_size = min(config.TEST_SIZE_A, config.TEST_SIZE_B)
    min_val_ds_size = min(config.VAL_SIZE_A, config.VAL_SIZE_B)
    config.METRIC_SAMPLE_SIZE_TRAIN = int(config.EVALUATE_PERCENT_OF_DATASET_TRAIN * min_train_ds_size / config.METRIC_BATCH_SIZE)
    config.METRIC_SAMPLE_SIZE_TEST = int(config.EVALUATE_PERCENT_OF_DATASET_TEST * min_test_ds_size / config.METRIC_BATCH_SIZE)
    config.METRIC_SAMPLE_SIZE_VAL = int(config.EVALUATE_PERCENT_OF_DATASET_VAL * min_val_ds_size / config.METRIC_BATCH_SIZE)
    config.EVALUATED_IMAGES_TRAIN = config.METRIC_SAMPLE_SIZE_TRAIN * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas
    config.EVALUATED_IMAGES_TEST = config.METRIC_SAMPLE_SIZE_TEST * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas 
    config.EVALUATED_IMAGES_VAL = config.METRIC_SAMPLE_SIZE_VAL * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas 

elif config.net_type == 'pix2pix':
    config.METRIC_SAMPLE_SIZE_TRAIN = int(config.EVALUATE_PERCENT_OF_DATASET_TRAIN * config.TRAIN_SIZE / config.METRIC_BATCH_SIZE)
    config.METRIC_SAMPLE_SIZE_TEST = int(config.EVALUATE_PERCENT_OF_DATASET_TEST * config.TEST_SIZE / config.METRIC_BATCH_SIZE)
    config.METRIC_SAMPLE_SIZE_VAL = int(config.EVALUATE_PERCENT_OF_DATASET_VAL * config.VAL_SIZE / config.METRIC_BATCH_SIZE)
    config.EVALUATED_IMAGES_TRAIN = config.METRIC_SAMPLE_SIZE_TRAIN * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas
    config.EVALUATED_IMAGES_TEST = config.METRIC_SAMPLE_SIZE_TEST * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas
    config.EVALUATED_IMAGES_VAL = config.METRIC_SAMPLE_SIZE_VAL * config.METRIC_BATCH_SIZE # Apenas para saber quantas imagens serão avaliadas


#%% FUNÇÕES DE TREINAMENTO

# FUNÇÕES DE TREINAMENTO DA CYCLEGAN

@tf.function
def train_step_cyclegan(gen_g, gen_f, disc_a, disc_b, real_a, real_b, 
                        gen_g_optimizer, gen_f_optimizer, disc_a_optimizer, disc_b_optimizer):

  with tf.GradientTape(persistent=True) as tape:
    # Gera as imagens falsas e as cicladas
    fake_b = gen_g(real_a, training=True)
    cycled_a = gen_f(fake_b, training=True)

    fake_a = gen_f(real_b, training=True)
    cycled_b = gen_g(fake_a, training=True)

    # Discrimina as imagens reais
    disc_real_a = disc_a(real_a, training=True)
    disc_real_b = disc_b(real_b, training=True)

    # Discrimina as imagens falsas
    disc_fake_a = disc_a(fake_a, training=True)
    disc_fake_b = disc_b(fake_b, training=True)

    # Losses adversárias de gerador
    gen_g_loss = generator_loss(disc_fake_b)
    gen_f_loss = generator_loss(disc_fake_a)
    
    # Loss de ciclo
    total_cycle_loss = cycle_loss(real_a, cycled_a) + cycle_loss(real_b, cycled_b)
    
    # Calcula a loss completa de gerador
    total_gen_g_loss = gen_g_loss + config.LAMBDA_CYCLEGAN * total_cycle_loss
    total_gen_f_loss = gen_f_loss + config.LAMBDA_CYCLEGAN * total_cycle_loss

    # Se precisar, adiciona a loss de identidade
    if config.USE_ID_LOSS:
        same_a = gen_f(real_a, training=True)
        same_b = gen_g(real_b, training=True)
        id_loss_b = identity_loss(real_b, same_b)
        id_loss_a = identity_loss(real_a, same_a)
        total_gen_g_loss += config.LAMBDA_CYCLEGAN * 0.5 * id_loss_b
        total_gen_f_loss += config.LAMBDA_CYCLEGAN * 0.5 * id_loss_a
    else:
        id_loss_a = 0
        id_loss_b = 0

    # Calcula as losses de discriminador
    disc_a_loss, disc_a_real_loss, disc_a_fake_loss = discriminator_loss(disc_real_a, disc_fake_a)
    disc_b_loss, disc_b_real_loss, disc_b_fake_loss = discriminator_loss(disc_real_b, disc_fake_b)
  
  # Calcula os gradientes
  gen_g_gradients = tape.gradient(total_gen_g_loss, gen_g.trainable_variables)
  gen_f_gradients = tape.gradient(total_gen_f_loss, gen_f.trainable_variables)
  
  disc_a_gradients = tape.gradient(disc_a_loss, disc_a.trainable_variables)
  disc_b_gradients = tape.gradient(disc_b_loss, disc_b.trainable_variables)
  
  # Realiza a atualização dos parâmetros através dos gradientes
  gen_g_optimizer.apply_gradients(zip(gen_g_gradients, gen_g.trainable_variables))
  gen_f_optimizer.apply_gradients(zip(gen_f_gradients, gen_f.trainable_variables))
  
  disc_a_optimizer.apply_gradients(zip(disc_a_gradients, disc_a.trainable_variables))
  disc_b_optimizer.apply_gradients(zip(disc_b_gradients, disc_b.trainable_variables))

  # Cria um dicionário das losses
  losses = {
      'gen_g_total_train': total_gen_g_loss,
      'gen_f_total_train': total_gen_f_loss,
      'gen_g_gan_train': gen_g_loss,
      'gen_f_gan_train': gen_f_loss,
      'cycle_loss_train': total_cycle_loss,
      'id_loss_a_train': id_loss_a,
      'id_loss_b_train': id_loss_b,
      'disc_a_total_train': disc_a_loss,
      'disc_b_total_train': disc_b_loss,
      'disc_a_real_train': disc_a_real_loss,
      'disc_b_real_train': disc_b_real_loss,
      'disc_a_fake_train': disc_a_fake_loss,
      'disc_b_fake_train': disc_b_fake_loss
  }

  return losses

def evaluate_validation_losses_cyclegan(gen_g, gen_f, disc_a, disc_b, real_a, real_b):

    # Gera as imagens falsas e as cicladas
    fake_b = gen_g(real_a, training=True)
    cycled_a = gen_f(fake_b, training=True)

    fake_a = gen_f(real_b, training=True)
    cycled_b = gen_g(fake_a, training=True)

    # Discrimina as imagens reais
    disc_real_a = disc_a(real_a, training=True)
    disc_real_b = disc_b(real_b, training=True)

    # Discrimina as imagens falsas
    disc_fake_a = disc_a(fake_a, training=True)
    disc_fake_b = disc_b(fake_b, training=True)

    # Losses adversárias de gerador
    gen_g_loss = generator_loss(disc_fake_b)
    gen_f_loss = generator_loss(disc_fake_a)
    
    # Loss de ciclo
    total_cycle_loss = cycle_loss(real_a, cycled_a) + cycle_loss(real_b, cycled_b)
    
    # Calcula a loss completa de gerador
    total_gen_g_loss = gen_g_loss + config.LAMBDA_CYCLEGAN * total_cycle_loss
    total_gen_f_loss = gen_f_loss + config.LAMBDA_CYCLEGAN * total_cycle_loss

    # Se precisar, adiciona a loss de identidade
    if config.USE_ID_LOSS:
        same_a = gen_f(real_a, training=True)
        same_b = gen_g(real_b, training=True)
        id_loss_b = identity_loss(real_b, same_b)
        id_loss_a = identity_loss(real_a, same_a)
        total_gen_g_loss += config.LAMBDA_CYCLEGAN * 0.5 * id_loss_b
        total_gen_f_loss += config.LAMBDA_CYCLEGAN * 0.5 * id_loss_a
    else:
        id_loss_a = 0
        id_loss_b = 0

    # Calcula as losses de discriminador
    disc_a_loss, disc_a_real_loss, disc_a_fake_loss = discriminator_loss(disc_real_a, disc_fake_a)
    disc_b_loss, disc_b_real_loss, disc_b_fake_loss = discriminator_loss(disc_real_b, disc_fake_b)

    # Cria um dicionário das losses
    losses = {
        'gen_g_total_val': total_gen_g_loss,
        'gen_f_total_val': total_gen_f_loss,
        'gen_g_gan_val': gen_g_loss,
        'gen_f_gan_val': gen_f_loss,
        'cycle_loss_val': total_cycle_loss,
        'id_loss_a_val': id_loss_a,
        'id_loss_b_val': id_loss_b,
        'disc_a_total_val': disc_a_loss,
        'disc_b_total_val': disc_b_loss,
        'disc_a_real_val': disc_a_real_loss,
        'disc_b_real_val': disc_b_real_loss,
        'disc_a_fake_val': disc_a_fake_loss,
        'disc_b_fake_val': disc_b_fake_loss
    }

    return losses

def fit_cyclegan(FIRST_EPOCH, EPOCHS, gen_g, gen_f, disc_a, disc_b,
                gen_g_optimizer, gen_f_optimizer, disc_a_optimizer, disc_b_optimizer):
  
    print("INICIANDO TREINAMENTO")

    # Prepara a progression bar
    qtd_smaller_dataset = config.TRAIN_SIZE_A if config.TRAIN_SIZE_A < config.TRAIN_SIZE_B else config.TRAIN_SIZE_B
    progbar_iterations = int(ceil(qtd_smaller_dataset / config.BATCH_SIZE))
    progbar = tf.keras.utils.Progbar(progbar_iterations)

    # Separa imagens fixas para acompanhar o treinamento
    for train_img_A, train_img_B, val_img_A, val_img_B in zip(
        train_A.take(1), train_B.take(1), val_A.shuffle(1).take(1), val_B.shuffle(1).take(1)):
        pass

    # Mostra como está a geração das imagens antes do treinamento
    utils.generate_fixed_images_cyclegan(train_img_A, train_img_B, val_img_A, val_img_B, gen_g, gen_f, FIRST_EPOCH -1, EPOCHS, result_folder, QUIET_PLOT)

    # Listas para o cálculo da acurácia
    y_real_A = []
    y_pred_A = []
    y_real_B = []
    y_pred_B = []

    for epoch in range(FIRST_EPOCH, EPOCHS+1):
        t1 = time.time()
        print("Época: ", epoch)

        # Train
        n = 0 # Contador da progression bar
        for image_a, image_b in tf.data.Dataset.zip((train_A, train_B)):

            # Faz o update da Progress Bar
            n += 1
            progbar.update(n)

            # Passo de treino
            losses_train = train_step_cyclegan(gen_g, gen_f, disc_a, disc_b, image_a, image_b, 
                                               gen_g_optimizer, gen_f_optimizer, disc_a_optimizer, disc_b_optimizer)
            
            # Cálculo da acurácia com imagens de validação
            y_real_A, y_pred_A, acc_A = metrics.evaluate_accuracy_cyclegan(gen_f, disc_a, val_A, y_real_A, y_pred_A)
            y_real_B, y_pred_B, acc_B = metrics.evaluate_accuracy_cyclegan(gen_g, disc_b, val_B, y_real_B, y_pred_B)
            losses_train['accuracy_A'] = acc_A
            losses_train['accuracy_B'] = acc_B

            # Acrescenta a época, para manter o controle
            losses_train['epoch'] = epoch

            # Loga as losses de treino no weights and biases
            wandb.log(utils.dict_tensor_to_numpy(losses_train))

            # A cada EVAL_ITERATIONS iterações, avalia as losses para o conjunto de teste
            if (n % config.EVAL_ITERATIONS) == 0 or n == 1 or n == progbar_iterations:
                for img_A, img_B in zip(val_A.unbatch().batch(config.BATCH_SIZE).take(1), val_B.unbatch().batch(config.BATCH_SIZE).take(1)):
                    losses_val = evaluate_validation_losses_cyclegan(gen_g, gen_f, disc_a, disc_b, img_A, img_B)

                    # Loga as losses de teste no weights and biases
                    wandb.log(utils.dict_tensor_to_numpy(losses_val))

        # Salva o checkpoint
        if config.SAVE_CHECKPOINT:
            if (epoch) % config.CHECKPOINT_EPOCHS == 0:
                ckpt_manager.save()
                print ('\nSalvando checkpoint da época {}'.format(epoch))

        # Gera as imagens após o treinamento desta época
        utils.generate_fixed_images_cyclegan(train_img_A, train_img_B, val_img_A, val_img_B, gen_g, gen_f, epoch, EPOCHS, result_folder, QUIET_PLOT)

        dt = time.time() - t1
        print ('\nTempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))
        
        # Loga o tempo de duração da época no wandb
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})

        ### AVALIAÇÃO DAS MÉTRICAS DE QUALIDADE ###
        if (config.EVALUATE_EVERY_EPOCH == True or
            config.EVALUATE_EVERY_EPOCH == False and epoch == EPOCHS):
            print("Avaliação das métricas de qualidade")

            if config.EVALUATE_TRAIN_IMGS:
                # Avaliação para as imagens de treino
                train_sample_A = train_A.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TRAIN) # Corrige o tamanho do batch
                train_sample_B = train_B.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TRAIN) # Corrige o tamanho do batch
                metric_results = metrics.evaluate_metrics_cyclegan(train_sample_A, train_sample_B, gen_g, gen_f, config.EVALUATE_IS, config.EVALUATE_FID)
                train_metrics = {k+"_train": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "_train" no final das keys
                wandb.log(train_metrics)

            # Avaliação para as imagens de validação
            val_sample_A = val_A.unbatch().shuffle(config.BUFFER_SIZE).batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_VAL) # Corrige o tamanho do batch
            val_sample_B = val_B.unbatch().shuffle(config.BUFFER_SIZE).batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_VAL) # Corrige o tamanho do batch
            metric_results = metrics.evaluate_metrics_cyclegan(val_sample_A, val_sample_B, gen_g, gen_f, config.EVALUATE_IS, config.EVALUATE_FID)
            val_metrics = {k+"_val": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "_val" no final das keys
            wandb.log(val_metrics)


# FUNÇÕES DE TREINAMENTO DA PIX2PIX

@tf.function
def train_step_pix2pix(gen, disc, gen_optimizer, disc_optimizer, input_img, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_img = gen(input_img, training=True)
          
        disc_real_output = disc([input_img, target], training=True)
        disc_fake_output = disc([input_img, fake_img], training=True)
          
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_pix2pix(disc_fake_output, fake_img, target)
        disc_loss, disc_real_loss, disc_fake_loss = discriminator_loss(disc_real_output, disc_fake_output)
    
    generator_gradients = gen_tape.gradient(gen_total_loss, gen.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, disc.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(generator_gradients, gen.trainable_variables))
    disc_optimizer.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    # Cria um dicionário das losses
    losses = {
        'gen_total_train': gen_total_loss,
        'gen_gan_train': gen_gan_loss,
        'gen_l1_train': gen_l1_loss,
        'disc_total_train': disc_loss,
        'disc_real_train': disc_real_loss,
        'disc_fake_train': disc_fake_loss,
    }
    
    return losses

def evaluate_validation_losses_pix2pix(gen, disc, input_img, target):
        
    fake_img = gen(input_img, training=True)

    disc_real_output = disc([input_img, target], training=True)
    disc_fake_output = disc([input_img, fake_img], training=True)
        
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_pix2pix(disc_fake_output, fake_img, target)
    disc_loss, disc_real_loss, disc_fake_loss = discriminator_loss(disc_real_output, disc_fake_output)

    # Cria um dicionário das losses
    losses = {
        'gen_total_val': gen_total_loss,
        'gen_gan_val': gen_gan_loss,
        'gen_l1_val': gen_l1_loss,
        'disc_total_val': disc_loss,
        'disc_real_val': disc_real_loss,
        'disc_fake_val': disc_fake_loss,
    }
    
    return losses

def fit_pix2pix(first_epoch, epochs, gen, disc, gen_optimizer, disc_optimizer):
    
    print("INICIANDO TREINAMENTO")

    # Prepara a progression bar
    progbar_iterations = int(ceil(config.TRAIN_SIZE / config.BATCH_SIZE))
    progbar = tf.keras.utils.Progbar(progbar_iterations)

    # Separa imagens fixas para acompanhar o treinamento
    for train_input, train_target in train_dataset.take(1):
        fixed_train = (train_input, train_target)
    for val_input, val_target in val_dataset.shuffle(1).take(1):
        fixed_val = (val_input,  val_target)

    # Mostra como está a geração das imagens antes do treinamento
    utils.generate_fixed_images_pix2pix(fixed_train, fixed_val, gen, first_epoch - 1, EPOCHS, result_folder, QUIET_PLOT)

    # Listas para o cálculo da acurácia
    y_real = []
    y_pred = []

    for epoch in range(first_epoch, epochs+1):
        t1 = time.time()
        print("Época: ", epoch)
        
        # Train
        for n, (input_image, target) in train_dataset.enumerate():

            # Faz o update da Progress Bar
            i = n.numpy() + 1 # Ajuste porque n começa em 0
            progbar.update(i)

            # Passo de treino
            losses_train = train_step_pix2pix(gen, disc, gen_optimizer, disc_optimizer, input_image, target)

            # Cálculo da acurácia com imagens de validação
            y_real, y_pred, acc = metrics.evaluate_accuracy_pix2pix(generator, disc, val_dataset, y_real, y_pred)
            losses_train['accuracy'] = acc

            # Acrescenta a época, para manter o controle
            losses_train['epoch'] = epoch

            # Loga as losses de treino no weights and biases
            wandb.log(utils.dict_tensor_to_numpy(losses_train))

            # A cada EVAL_ITERATIONS iterações, avalia as losses para o conjunto de val
            if (n % config.TEST_EVAL_ITERATIONS) == 0 or n == 1 or n == progbar_iterations:
                for example_input, example_target in val_dataset.unbatch().batch(config.BATCH_SIZE).take(1):
                    # Calcula as losses
                    losses_val = evaluate_validation_losses_pix2pix(gen, disc, example_input, example_target)
                    # Loga as losses de val no weights and biases
                    wandb.log(utils.dict_tensor_to_numpy(losses_val))
        
        # Salva o checkpoint
        if config.SAVE_CHECKPOINT:
            if (epoch) % config.CHECKPOINT_EPOCHS == 0:
                ckpt_manager.save()
                print ('\nSalvando checkpoint da época {}'.format(epoch))

        # Gera as imagens após o treinamento desta época
        utils.generate_fixed_images_pix2pix(fixed_train, fixed_val, gen, epoch, EPOCHS, result_folder, QUIET_PLOT)

        dt = time.time() - t1
        print ('Tempo usado para a época {} foi de {:.2f} min ({:.2f} sec)\n'.format(epoch, dt/60, dt))

        # Loga o tempo de duração da época no wandb
        wandb.log({'epoch time (s)': dt, 'epoch time (min)': dt/60})
        
        ### AVALIAÇÃO DAS MÉTRICAS DE QUALIDADE ###
        if (config.EVALUATE_EVERY_EPOCH == True or
            config.EVALUATE_EVERY_EPOCH == False and epoch == EPOCHS):
            print("Avaliação das métricas de qualidade")

            if config.EVALUATE_TRAIN_IMGS:
                # Avaliação para as imagens de treino
                train_sample = train_dataset.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TRAIN) # Corrige o tamanho do batch
                metric_results = metrics.evaluate_metrics_pix2pix(train_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
                train_metrics = {k+"_train": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "train" no final das keys
                wandb.log(train_metrics)

            # Avaliação para as imagens de validação
            val_sample = val_dataset.unbatch().shuffle(config.BUFFER_SIZE).batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_VAL) # Corrige o tamanho do batch
            metric_results = metrics.evaluate_metrics_pix2pix(val_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
            val_metrics = {k+"_val": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "val" no final das keys
            wandb.log(val_metrics)


#%% PREPARAÇÃO DOS MODELOS

# ---- GERADORES
if config.net_type == 'cyclegan':
    if config.gen_model == 'pix2pix':
        generator_g = networks.Pix2Pix_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS)
        generator_f = networks.Pix2Pix_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS)
    elif config.gen_model == 'unet':
        generator_g = networks.Unet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
        generator_f = networks.Unet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    elif config.gen_model == 'cyclegan':
        generator_g = networks.CycleGAN_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm', num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
        generator_f = networks.CycleGAN_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm', num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
    else:
        raise utils.GeneratorError(config.gen_model)

elif config.net_type == 'pix2pix':
    if config.gen_model == 'pix2pix':
        generator = networks.Pix2Pix_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS)
    elif config.gen_model == 'unet':
        generator = networks.Unet_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    elif config.gen_model == 'cyclegan':
        generator = networks.CycleGAN_Generator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm', num_residual_blocks=config.NUM_RESIDUAL_BLOCKS)
    else:
        raise utils.GeneratorError(config.gen_model)

else:
    raise utils.ArchitectureError(config.net_type)


# ---- DISCRIMINADORES
if config.net_type == 'cyclegan':
    discriminator_a = networks.CycleGAN_Discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')
    discriminator_b = networks.CycleGAN_Discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')

elif config.net_type == 'pix2pix':
    discriminator = networks.Pix2Pix_Discriminator(config.IMG_SIZE, config.OUTPUT_CHANNELS, norm_type='instancenorm')

else:
    raise utils.ArchitectureError(config.net_type)


# ---- OTIMIZADORES
if config.net_type == 'cyclegan':
    generator_g_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    generator_f_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    discriminator_a_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    discriminator_b_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)

elif config.net_type == 'pix2pix':
    generator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)
    discriminator_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1 = config.ADAM_BETA_1)

else:
    raise utils.ArchitectureError(config.net_type)

#%% CHECKPOINTS

if config.net_type == 'cyclegan':
    ckpt = tf.train.Checkpoint(generator_g = generator_g,
                            generator_f = generator_f,
                            discriminator_a = discriminator_a,
                            discriminator_b = discriminator_b,
                            generator_g_optimizer = generator_g_optimizer,
                            generator_f_optimizer = generator_f_optimizer,
                            discriminator_a_optimizer = discriminator_a_optimizer,
                            discriminator_b_optimizer = discriminator_b_optimizer)

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
    else:
        FIRST_EPOCH = config.FIRST_EPOCH

#%% TREINAMENTO

if config.TRAIN:
    if config.net_type == 'cyclegan':
        fit_cyclegan(FIRST_EPOCH, EPOCHS, generator_g, generator_f, discriminator_a, discriminator_b,
                    generator_g_optimizer, generator_f_optimizer, discriminator_a_optimizer, discriminator_b_optimizer)

    elif config.net_type == 'pix2pix':
        fit_pix2pix(FIRST_EPOCH, EPOCHS, generator, discriminator, generator_optimizer, discriminator_optimizer)

    else:
        raise utils.ArchitectureError(config.net_type)

#%% VALIDAÇÃo

if config.VALIDATION:

    # Gera imagens do dataset de validação

    if config.net_type == 'cyclegan':
        print("\nCriando imagens do conjunto de validação...")

        ## -- A TO B --
        print("\nA to B")

        # Caso seja -1, plota tudo
        if config.NUM_VAL_PRINTS < 0:
            num_imgs = config.VAL_SIZE_A
        else:
            num_imgs = config.NUM_VAL_PRINTS
        
        for c, inp in val_A.enumerate():
            # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
            if config.NUM_VAL_PRINTS >= 0 and c >= config.NUM_VAL_PRINTS:
                break
            # Rotina de plot das imagens de validação
            c = c.numpy()
            onepercent = int(num_imgs / 100) if int(num_imgs / 100) !=0 else 1 # Evita Div 0
            if c % onepercent == 0 or c == 0 or c == num_imgs:
                print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c+1, num_imgs, 100*(c+1)/num_imgs))
            filename = "A_to_B_val_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
            utils.generate_images_cyclegan(generator_g, inp, result_val_folder, filename, quiet = QUIET_PLOT)

        ## -- B TO A --
        print("\nB to A")

        # Caso seja -1, plota tudo
        if config.NUM_VAL_PRINTS < 0:
            num_imgs = config.VAL_SIZE_B
        else:
            num_imgs = config.NUM_VAL_PRINTS
        
        for c, inp in val_B.enumerate():
            # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
            if config.NUM_VAL_PRINTS >= 0 and c >= config.NUM_VAL_PRINTS:
                break
            # Rotina de plot das imagens de validação
            c = c.numpy()
            onepercent = int(num_imgs / 100) if int(num_imgs / 100) !=0 else 1 # Evita Div 0
            if c % onepercent == 0 or c == 0 or c == num_imgs:
                print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c+1, num_imgs, 100*(c+1)/num_imgs))
            filename = "B_to_A_val_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
            utils.generate_images_cyclegan(generator_f, inp, result_val_folder, filename, quiet = QUIET_PLOT)

    elif config.net_type == 'pix2pix':
        print("\nCriando imagens do conjunto de validação...")

        # Caso seja -1, plota tudo
        if config.NUM_VAL_PRINTS < 0:
            num_imgs = config.val_SIZE
        else:
            num_imgs = config.NUM_VAL_PRINTS

        for c, (inp, tar) in val_dataset.enumerate():
            # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
            if config.NUM_VAL_PRINTS >= 0 and c >= config.NUM_VAL_PRINTS:
                break
            # Rotina de plot das imagens de vale
            c = c.numpy()
            onepercent = int(num_imgs / 100) if int(num_imgs / 100) !=0 else 1 # Evita Div 0
            if c % onepercent == 0 or c == 0 or c == num_imgs:
                print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c+1, num_imgs, 100*(c+1)/num_imgs))
            filename = "val_results_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
            utils.generate_images_pix2pix(generator, inp, tar, result_val_folder, filename, quiet = QUIET_PLOT)

    else:
        raise utils.ArchitectureError(config.net_type)


#%% TESTE

if config.TEST:

    # Gera imagens do dataset de teste

    if config.net_type == 'cyclegan':
        print("\nCriando imagens do conjunto de teste...")

        ## -- A TO B --
        print("\nA to B")
        t1 = time.time()

        # Caso seja -1, plota tudo
        if config.NUM_TEST_PRINTS < 0:
            num_imgs = config.TEST_SIZE_A
        else:
            num_imgs = config.NUM_TEST_PRINTS
        
        for c, inp in test_A.enumerate():
            # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
            if config.NUM_TEST_PRINTS >= 0 and c >= config.NUM_TEST_PRINTS:
                break
            # Rotina de plot das imagens de teste
            c = c.numpy()
            onepercent = int(num_imgs / 100) if int(num_imgs / 100) !=0 else 1 # Evita Div 0
            if c % onepercent == 0 or c == 0 or c == num_imgs:
                print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c+1, num_imgs, 100*(c+1)/num_imgs))
            filename = "A_to_B_test_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
            utils.generate_images_cyclegan(generator_g, inp, result_test_folder, filename, quiet = QUIET_PLOT)
        
        dt = time.time() - t1

        if num_imgs != 0:
            mean_inference_time_A = dt / num_imgs

        ## -- B TO A --
        print("\nB to A")
        t1 = time.time()

        # Caso seja -1, plota tudo
        if config.NUM_TEST_PRINTS < 0:
            num_imgs = config.TEST_SIZE_B
        else:
            num_imgs = config.NUM_TEST_PRINTS
        
        for c, inp in test_B.enumerate():
            # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
            if config.NUM_TEST_PRINTS >= 0 and c >= config.NUM_TEST_PRINTS:
                break
            # Rotina de plot das imagens de teste
            c = c.numpy()
            onepercent = int(num_imgs / 100) if int(num_imgs / 100) !=0 else 1 # Evita Div 0
            if c % onepercent == 0 or c == 0 or c == num_imgs:
                print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c+1, num_imgs, 100*(c+1)/num_imgs))
            filename = "B_to_A_test_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
            utils.generate_images_cyclegan(generator_f, inp, result_test_folder, filename, quiet = QUIET_PLOT)

        dt = time.time() - t1

        if num_imgs != 0:
            mean_inference_time_B = dt / num_imgs

            # Loga os tempos de inferência no wandb
            wandb.log({'mean inference time A (s)': mean_inference_time_A, 'mean inference time B (s)': mean_inference_time_B})
    
    elif config.net_type == 'pix2pix':
        print("\nCriando imagens do conjunto de teste...")
        t1 = time.time()

        # Caso seja -1, plota tudo
        if config.NUM_TEST_PRINTS < 0:
            num_imgs = config.TEST_SIZE
        else:
            num_imgs = config.NUM_TEST_PRINTS

        for c, (inp, tar) in test_dataset.enumerate():
            # Caso seja zero, não plota nenhuma. Caso seja um número positivo, plota a quantidade descrita.
            if config.NUM_TEST_PRINTS >= 0 and c >= config.NUM_TEST_PRINTS:
                break
            # Rotina de plot das imagens de teste
            c = c.numpy()
            onepercent = int(num_imgs / 100) if int(num_imgs / 100) !=0 else 1 # Evita Div 0
            if c % onepercent == 0 or c == 0 or c == num_imgs:
                print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c+1, num_imgs, 100*(c+1)/num_imgs))
            filename = "test_results_" + str(c+1).zfill(len(str(num_imgs))) + ".jpg"
            utils.generate_images_pix2pix(generator, inp, tar, result_test_folder, filename, quiet = QUIET_PLOT)

        dt = time.time() - t1

        if num_imgs != 0:
            mean_inference_time = dt / num_imgs

            # Loga os tempos de inferência no wandb
            wandb.log({'mean inference time (s)': mean_inference_time})
    
    else:
        raise utils.ArchitectureError(config.net_type)

    # Gera métricas do dataset de teste

    if config.net_type == 'cyclegan':
        print("Iniciando avaliação das métricas de qualidade do dataset de teste")
        test_sample_A = test_A.unbatch().shuffle(config.BUFFER_SIZE).batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TEST) # Corrige o tamanho do batch
        test_sample_B = test_B.unbatch().shuffle(config.BUFFER_SIZE).batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_TEST) # Corrige o tamanho do batch
        metric_results = metrics.evaluate_metrics_cyclegan(test_sample_A, test_sample_B, generator_g, generator_f, config.EVALUATE_IS, config.EVALUATE_FID)
        test_metrics = {k+"_test": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "_test" no final das keys
        wandb.log(test_metrics)

    elif config.net_type == 'pix2pix':
        print("Iniciando avaliação das métricas de qualidade do dataset de teste")
        test_sample = test_dataset.unbatch().batch(config.METRIC_BATCH_SIZE).take(config.METRIC_SAMPLE_SIZE_test) # Corrige o tamanho do batch
        metric_results = metrics.evaluate_metrics_pix2pix(test_sample, generator, config.EVALUATE_IS, config.EVALUATE_FID, config.EVALUATE_L1)
        test_metrics = {k+"_test": v for k, v in metric_results.items()} # Renomeia o dicionário para incluir "_test" no final das keys
        wandb.log(test_metrics)
     
    else:
        raise utils.ArchitectureError(config.net_type)

#%% GENERALIZAÇÃO

if config.GENERALIZATION and config.net_type == 'cyclegan' and dataset_folder == cars_folder: 
    
    print("\nINICIANDO TESTE DE GENERALIZAÇÃO")
    
    generalization_read_prefix = base_root
    generalization_read_folder_A = generalization_read_prefix + 'generalization_images_car_cycle/classA/'
    generalization_read_folder_B = generalization_read_prefix + 'generalization_images_car_cycle/classB/'

    generalization_save_prefix = experiment_folder + 'generalization_results/'    
    generalization_save_folder_A = generalization_save_prefix + 'AtoB/'
    generalization_save_folder_B = generalization_save_prefix + 'BtoA/'
    
    if not os.path.exists(generalization_save_prefix):
        os.mkdir(generalization_save_prefix)
    
    if not os.path.exists(generalization_save_folder_A):
        os.mkdir(generalization_save_folder_A)
    
    if not os.path.exists(generalization_save_folder_B):
        os.mkdir(generalization_save_folder_B)
    
    ## GENERALIZAÇÃO A TO B
    
    # Encontra os arquivos:
    files = [f for f in os.listdir(generalization_read_folder_A) if os.path.isfile(os.path.join(generalization_read_folder_A, f))]
    ds_size = len(files)
    print("Encontrado {0} arquivos".format(ds_size))
    
    c = 1
    for file in files:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, ds_size, 100*c/ds_size))
        filepath = generalization_read_folder_A + file
        image = utils.generalization_load_A(filepath, config.IMG_SIZE)
        image = np.expand_dims(image, axis=0)
            
        filename = "generalization_results_" + str(c).zfill(len(str(ds_size))) + ".jpg"
        utils.generate_images_cyclegan(generator_g, image, generalization_save_folder_A, filename, quiet = QUIET_PLOT)
        c = c + 1
    
    ## GENERALIZAÇÃO B TO A
    
    # Encontra os arquivos:
    files = [f for f in os.listdir(generalization_read_folder_B) if os.path.isfile(os.path.join(generalization_read_folder_B, f))]
    ds_size = len(files)
    print("Encontrado {0} arquivos".format(ds_size))
    
    c = 1
    for file in files:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, ds_size, 100*c/ds_size))
        filepath = generalization_read_folder_B + file
        image = utils.generalization_load_B(filepath, config.IMG_SIZE)
        image = np.expand_dims(image, axis=0)
            
        filename = "generalization_results_" + str(c).zfill(len(str(ds_size))) + ".jpg"
        utils.generate_images_cyclegan(generator_f, image, generalization_save_folder_B, filename, quiet = QUIET_PLOT)
        c = c + 1
    
if config.GENERALIZATION and config.net_type == 'pix2pix' and dataset_folder == cars_paired_folder or dataset_folder == cars_paired_folder_complete:

    print("\nINICIANDO GENERALIZAÇÃO")

    generalization_read_prefix = base_root
    generalization_read_folder = generalization_read_prefix + 'generalization_images_car_sketches/'

    generalization_save_prefix = experiment_folder   
    generalization_save_folder = generalization_save_prefix + 'generalization_results/'

    if not os.path.exists(generalization_save_folder):
        os.mkdir(generalization_save_folder)

    # Encontra os arquivos:
    files = [f for f in os.listdir(generalization_read_folder) if os.path.isfile(os.path.join(generalization_read_folder, f))]
    ds_size = len(files)
    print("Encontrado {0} arquivos".format(ds_size))

    c = 1
    for file in files:
        
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, ds_size, 100*c/ds_size))
        
        filepath = generalization_read_folder + file
        image = utils.generalization_load_B(filepath, config.IMG_SIZE)

        image = np.expand_dims(image, axis=0)
            
        filename = "generalization_results_" + str(c).zfill(len(str(ds_size))) + ".jpg"
        utils.generate_images_pix2pix(generator, image, image, generalization_save_folder, filename, quiet = QUIET_PLOT)
        
        c = c + 1

#%% TESTE DE CICLAGEM

if config.CYCLE_TEST and config.net_type == 'cyclegan':
    
    print("\nINICIANDO TESTE DE MÚLTIPLAS CICLAGENS")
    
    cycle_save_folder = experiment_folder + 'cycle_test/'
    AtoB_folder = cycle_save_folder + 'AtoB/'
    BtoA_folder = cycle_save_folder + 'BtoA/'
    
    if not os.path.exists(cycle_save_folder):
        os.mkdir(cycle_save_folder)

    if not os.path.exists(AtoB_folder):
        os.mkdir(AtoB_folder)

    if not os.path.exists(BtoA_folder):
        os.mkdir(BtoA_folder)

    # Realiza o teste de ciclagem A -> B
    mean_l1_distance_AtoB = utils.cycle_test(test_A, generator_g, generator_f, config.CYCLES, config.CYCLE_TEST_PICTURES, AtoB_folder, start = 'A', QUIET_PLOT = QUIET_PLOT)
    print("Erro médio de reconstrução de ciclo A -> B: {}".format(mean_l1_distance_AtoB))
    # Loga no wandb
    wandb.log({'l1_cycle_AtoB': mean_l1_distance_AtoB})

    # Realiza o teste de ciclagem B -> A
    mean_l1_distance_BtoA = utils.cycle_test(test_B, generator_f, generator_g, config.CYCLES, config.CYCLE_TEST_PICTURES, BtoA_folder, start = 'B', QUIET_PLOT = QUIET_PLOT)
    print("Erro médio de reconstrução de ciclo B -> A: {}".format(mean_l1_distance_BtoA))
    # Loga no wandb
    wandb.log({'l1_cycle_BtoA': mean_l1_distance_BtoA})

#%% FINAL

# Finaliza o Weights and Biases
wandb.finish()

## Salva os modelos
if config.SAVE_MODELS and config.TRAIN:
    print("Salvando modelos...\n")

    if config.net_type == 'cyclegan':
        generator_g.save_weights(model_folder+'generator_g.tf')
        generator_f.save_weights(model_folder+'generator_f.tf')
        discriminator_a.save_weights(model_folder+'discriminator_a.tf')
        discriminator_b.save_weights(model_folder+'discriminator_b.tf')
    elif config.net_type == 'pix2pix':
        generator.save(model_folder+'generator.h5')
        discriminator.save(model_folder+'discriminator.h5')

## Desliga o PC ao final do processo, se for o caso
if SHUTDOWN_AFTER_FINISH:
    time.sleep(60)
    os.system("shutdown /s /t 10")