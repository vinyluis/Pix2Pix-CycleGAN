# Classificador binário Carro vs Não Carro

import os
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import numpy as np
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

from tensorflow.data import Dataset
from tensorflow.keras.preprocessing import image_dataset_from_directory as image_dataset

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Evita o erro "Failed to get convolution algorithm. This is probably because cuDNN failed to initialize"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Verifica se a GPU está disponível:
print(tf.config.list_physical_devices('GPU'))
# Verifica se a GPU está sendo usada na sessão
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(sess)


# Prepara a pasta
PATH = '60k_car_classifier/'
result_folder = '60k_car_dataset_selected/'

if not os.path.exists(result_folder):
    os.mkdir(result_folder)

# HIPERPARÂMETROS
BATCH_SIZE = 10
IMG_SIZE = 256
NUMBER_CHANNELS = 3
EPOCHS = 50
LEARNING_RATE = 0.0002
TEST_RATIO = 0.2
TRESH = 0.5


#%% FUNÇÕES

# Define o classificador
def classifier(input_shape):
    model = Sequential()
    
    model.add(Conv2D(8, (3,3), strides=(2,2), padding = 'same', input_shape = input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
	
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3,3), strides=(2,2), padding = 'same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3), strides=(2,2), padding = 'same', activation='relu'))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile model
    opt = Adam(lr=LEARNING_RATE, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    return model


# Rotina de treino
def train(model, dataset, epochs = 50, batch_size = 128):
    
    losses = []
    for epoch in range(epochs):
        print("Epoch {} / {}".format(epoch+1, epochs))
        
        for images, labels in list(dataset.as_numpy_iterator()):
            loss = model.train_on_batch(images, labels)
            losses.append(loss[0])
            
    # save the generator model
    model.save('classifier.h5')
    return model, losses


# Avaliação do classificador
def eval_classifier(real, predicted):
    cm = confusion_matrix(real, predicted)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    print("Resultados:")
    print("\n Matriz de Confusão:")
    print(cm)
    
    # Accuracy
    acc = (TN + TP)/cm.sum()
    print("\nAcurácia: {0:.2f}%".format(100*acc))
    
    # Precision and Recall
    # Precision - What proportion of positive identifications was actually correct?
    precision = TP / (TP + FP)
    print("Precisão (De todas as previsões '1', quantas acertaram): {0:.2f}%".format(100 * precision))
    # Recall - What proportion of actual positives was identified correctly?
    recall = TP / (TP + FN)
    print("Recall (De todos os valores que são verdadeiramente '1', quanto acertei): {0:.2f}%".format(100 * recall))
    
    # Area Under ROC Curve (auc)
    auc = roc_auc_score(real, predicted)
    print("Area Under ROC Curve: {0:.4f}".format(auc))
    
    print("")
    
    # Create result dictionary
    res = {}
    res["accuracy"] = acc
    res["confusion_matrix"] = cm
    res["precision"] = precision
    res['recall'] = recall
    res['auc'] = auc
    
    return res
    
#%% Carrega o dataset

random_seed = int(np.random.rand()*100)
dataset_train = image_dataset(PATH, batch_size = BATCH_SIZE, validation_split = TEST_RATIO, 
                              seed = random_seed, shuffle = True,  subset = 'training',
                              image_size = (IMG_SIZE, IMG_SIZE))

dataset_test = image_dataset(PATH, batch_size = 1, validation_split = TEST_RATIO, 
                             seed = random_seed, shuffle = True,  subset = 'validation', 
                             image_size = (IMG_SIZE, IMG_SIZE))


#%% Cria e treina o modelo
 
print("\nIniciando o treinamento")
model = classifier((IMG_SIZE, IMG_SIZE, NUMBER_CHANNELS))
tf.keras.utils.plot_model(model, show_shapes=True, dpi=64)

model, losses = train(model, dataset_train, EPOCHS, BATCH_SIZE)

sns.lineplot(x = range(len(losses)), y = losses)

#%% Testa a classificação

print("\nValidando o modelo \n")

y_pred = []
y_real = []
for image, label in dataset_test:
    
    pred = model.predict(image)   
    pred = 1 if pred > 0.5 else 0
    
    y_pred.append(pred)
    y_real.append(label.numpy()[0])
    
    
result = eval_classifier(y_real, y_pred)


#%% Cria o dataset final
'''
################################################
# CUIDADO! ESSE CÓDIGO VAI RODAR EM 64K IMAGENS#
################################################

print("\nSeparando o dataset final")

DATA = '60k_car_dataset/'

# Encontra os arquivos:
files = [f for f in listdir(DATA) if isfile(join(DATA, f))]
num_files = len(files)
print("Encontrado {0} arquivos".format(num_files))


c = 1
for file in files:
    
    if c % 100 == 0 or c == num_files or c==1:
        print("[{0:5d} / {1:5d}] {2:5.2f}%".format(c, num_files, 100*c/num_files))
    
    filepath = DATA + file
    
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = np.expand_dims(image, axis=0)
    
    pred = model.predict(image)   
    pred = 1 if pred > 0.5 else 0
    
    if pred == 0:
        new_filepath = result_folder + str(c).zfill(len(str(num_files))) + ".jpg"
        copyfile(filepath, new_filepath)
    
    c = c + 1
    
    #if c >= 10:
    #    break
'''