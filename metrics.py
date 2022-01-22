import os
import time
import numpy as np

from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score as accuracy
from tensorflow.python.framework.tensor_shape import as_dimension

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Silencia o TF (https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information)
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

import utils


#%% PREPARAÇÃO

# Prepara o modelo Inception v3 para o IS
model_IS = InceptionV3()

# Prepara o modelo Inception v3 para o FID
model_FID = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

#%% FUNÇÕES BASE

def evaluate_metrics_cyclegan(sample_ds_A, sample_ds_B, generator_g, generator_f, evaluate_is, evaluate_fid, verbose = False):
	"""Calcula as métricas de qualidade para o framework CycleGAN.

	Calcula Inception Score e Frechét Inception Distance para todos os geradores.
	"""

	t1 = time.time()
	inception_score_A = []
	inception_score_B = []
	frechet_inception_distance_A = []
	frechet_inception_distance_B = []
	c = 0
	for input_A, input_B in tf.data.Dataset.zip((sample_ds_A, sample_ds_B)):
		
		c += 1
		if verbose:
			print(c)

		# Para cada imagem, calcula sua versão sintética
		fake_B = generator_g(input_A)
		fake_A = generator_f(input_B)
	
		try:
			# Cálculos da IS
			if evaluate_is:
				is_score_A = get_inception_score(fake_A)
				is_score_B = get_inception_score(fake_B)
				inception_score_A.append(is_score_A)
				inception_score_B.append(is_score_B)
				if verbose: 
					print("IS (A) = {:.2f}".format(is_score_A))
					print("IS (B) = {:.2f}".format(is_score_B))

			# Cálculos da FID
			if evaluate_fid:
				fid_score_A = get_frechet_inception_distance(fake_A, input_A)
				fid_score_B = get_frechet_inception_distance(fake_B, input_B)
				frechet_inception_distance_A.append(fid_score_A)
				frechet_inception_distance_B.append(fid_score_B)
				if verbose: 
					print("FID (A) = {:.2f}".format(fid_score_A))
					print("FID (B) = {:.2f}".format(fid_score_B))

		except:
			if verbose:
				print("Erro na {}-ésima iteração. Pulando.".format(c))

		if verbose:
			print()

	# Calcula os scores consolidados e salva em um dicionário
	results = {}
	if evaluate_is:
		is_avg_A, is_std_A = np.mean(inception_score_A), np.std(inception_score_A)
		results['is_avg_A'] = is_avg_A
		results['is_std_A'] = is_std_A
		is_avg_B, is_std_B = np.mean(inception_score_B), np.std(inception_score_B)
		results['is_avg_B'] = is_avg_B
		results['is_std_B'] = is_std_B
	if evaluate_fid:
		fid_avg_A, fid_std_A = np.mean(frechet_inception_distance_A), np.std(frechet_inception_distance_A)
		results['fid_avg_A'] = fid_avg_A
		results['fid_std_A'] = fid_std_A
		fid_avg_B, fid_std_B = np.mean(frechet_inception_distance_B), np.std(frechet_inception_distance_B)
		results['fid_avg_B'] = fid_avg_B
		results['fid_std_B'] = fid_std_B

	# Reporta o resultado
	if verbose:
		if evaluate_is:
			print("Inception Score A:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(is_avg_A, is_std_A))
			print("Inception Score B:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(is_avg_B, is_std_B))
		if evaluate_fid:
			print("Fréchet Inception Distance A:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(fid_avg_A, fid_std_A))
			print("Fréchet Inception Distance B:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(fid_avg_B, fid_std_B))

	dt = time.time() - t1
	results['eval_time'] = dt

	return results

def evaluate_metrics_pix2pix(sample_ds, generator, evaluate_is, evaluate_fid, evaluate_l1, verbose = False):
	"""Calcula as métricas de qualidade para o framework Pix2Pix.

	Calcula Inception Score e Frechét Inception Distance para o gerador.
	Calcula a distância L1 (distância média absoluta pixel a pixel) entre a imagem sintética e a objetivo.
	"""
	t1 = time.time()
	inception_score = []
	frechet_inception_distance = []
	l1_distance = []
	c = 0
	for input_img, target in sample_ds:
		
		c += 1
		if verbose:
			print(c)

		# Para cada imagem, calcula sua versão sintética
		fake = generator(input_img)

		try:
			# Cálculos da IS
			if evaluate_is:
				is_score = get_inception_score(fake)
				inception_score.append(is_score)
				if verbose: 
					print("IS = {:.2f}".format(is_score))

			# Cálculos da FID
			if evaluate_fid:
				fid_score = get_frechet_inception_distance(fake, target)
				frechet_inception_distance.append(fid_score)
				if verbose: 
					print("FID = {:.2f}".format(fid_score))

			# Cálculos da L1
			if evaluate_l1:
				l1_score = get_l1_distance(fake, target)
				l1_distance.append(l1_score)
				if verbose: 
					print("L1 = {:.2f}".format(l1_score))

		except:
			if verbose:
				print("Erro na {}-ésima iteração. Pulando.".format(c))

		if verbose:
			print()

	# Calcula os scores consolidados e salva em um dicionário
	results = {}
	if evaluate_is:
		is_avg, is_std = np.mean(inception_score), np.std(inception_score)
		results['is_avg'] = is_avg
		results['is_std'] = is_std
	if evaluate_fid:
		fid_avg, fid_std = np.mean(frechet_inception_distance), np.std(frechet_inception_distance)
		results['fid_avg'] = fid_avg
		results['fid_std'] = fid_std
	if evaluate_l1:
		l1_avg, l1_std = np.mean(l1_distance), np.std(l1_distance)
		results['l1_avg'] = l1_avg
		results['l1_std'] = l1_std

	# Reporta o resultado
	if verbose:
		if evaluate_is:
			print("Inception Score:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(is_avg, is_std))
		if evaluate_fid:
			print("Fréchet Inception Distance:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(fid_avg, fid_std))
		if evaluate_l1:
			print("L1 Distance:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(l1_avg, l1_std))

	dt = time.time() - t1
	results['eval_time'] = dt

	return results


#%% FUNÇÕES DE CÁLCULO DAS MÉTRICAS

# Inception Score
def get_inception_score(image):
	'''
	Calcula o Inception Score (IS) para uma única imagem. 
	Baseado em: https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
	'''
	# Epsilon para evitar problemas no cálculo da divergência KL
	eps=1E-16
	# Redimensiona a imagem
	image = utils.resize(image, 299, 299)
	# Usa o Inception v3 para calcular a probabilidade condicional p(y|x)
	p_yx = model_IS.predict(image)
	# Calcula p(y)
	p_y = np.expand_dims(p_yx.mean(axis=0), 0)
	# Calcula a divergência KL usando probabilididades log
	kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
	# Soma todas as classes da inception
	sum_kl_d = kl_d.sum(axis=1)
	# Faz a média para a imagem
	avg_kl_d = np.mean(sum_kl_d)
	# Desfaz o log
	is_score = np.exp(avg_kl_d)
	
	return is_score

# Frechet Inception Distance
def get_frechet_inception_distance(image1, image2):
	'''
	Calcula o Fréchet Inception Distance (FID) entre duas imagens. 
	Baseado em: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
	'''
	# Redimensiona as imagens
	image1 = utils.resize(image1, 299, 299)
	image2 = utils.resize(image2, 299, 299)
	# Calcula as ativações
	act1 = model_FID.predict(image1)
	act2 = model_FID.predict(image2)
	# Calcula as estatísticas de média (mu) e covariância (sigma)
	mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
	# Calcula a distância L2 das médias
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# Calcula a raiz do produto entre as matrizes de covariância
	covmean = sqrtm(sigma1.dot(sigma2))
	# Corrige números imaginários, se necessário
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# Calcula o score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	
	return fid

# L1 Distance
def get_l1_distance(image1, image2):
	'''Calcula a distância L1 (distância média absoluta pixel a pixel) entre duas imagens'''
	l1_dist = tf.reduce_mean(tf.abs(image1 - image2))
	return l1_dist

# Avaliação de acurácia do discriminador - Pix2Pix
def evaluate_accuracy_pix2pix(gen, disc, test_ds, y_real, y_pred, window = 100):
	'''Avalia a acurácia do discriminador no framework Pix2Pix.

	Compara o discriminador a um classificador binário.
	'''
	# Gera uma imagem-base
	for img_real, target in test_ds.take(1):
		
		# A partir dela, gera uma imagem sintética
		img_fake = gen(img_real, training = True)

		# Avalia ambas
		disc_real = disc([img_real, target], training = True)
		disc_fake = disc([img_fake, target], training = True)

		# Para o caso de ser um discriminador PatchGAN, tira a média
		disc_real = np.mean(disc_real)
		disc_fake = np.mean(disc_fake)

		# Aplica o threshold
		disc_real = 1 if disc_real > 0.5 else 0
		disc_fake = 1 if disc_fake > 0.5 else 0

		# Acrescenta a observação real como y_real = 1
		y_real.append(1)
		y_pred.append(disc_real)

		# Acrescenta a observação fake como y_real = 0
		y_real.append(0)
		y_pred.append(disc_fake)

		# Calcula a acurácia pela janela
		if len(y_real) > window:
			acc = accuracy(y_real[-window:], y_pred[-window:])    
		else:
			acc = accuracy(y_real, y_pred)

	return y_real, y_pred, acc

# Avaliação de acurácia do discriminador - CycleGAN
def evaluate_accuracy_cyclegan(gen, disc, test_ds, y_real, y_pred, window = 100):
	'''Avalia a acurácia do discriminador no framework CycleGAN.
	
	Gerador G: A->B
	Gerador F: B->A
	Discriminador A: Discrimina A
	Discriminador B: Discrimina B
	'''

	# Gera uma imagem-base
	for img_real in test_ds.take(1):

		# A partir dela, gera uma imagem sintética
		img_fake = gen(img_real, training = True)

		# Avalia ambas
		disc_real = disc(img_real, training = True)
		disc_fake = disc(img_fake, training = True)

		# Para o caso de ser um discriminador PatchGAN, tira a média
		disc_real = np.mean(disc_real)
		disc_fake = np.mean(disc_fake)

		# Aplica o threshold
		disc_real = 1 if disc_real > 0.5 else 0
		disc_fake = 1 if disc_fake > 0.5 else 0

		# Acrescenta a observação real como y_real = 1
		y_real.append(1)
		y_pred.append(disc_real)

		# Acrescenta a observação fake como y_real = 0
		y_real.append(0)
		y_pred.append(disc_fake)
		
		# Calcula a acurácia pela janela
		if len(y_real) > window:
			acc = accuracy(y_real[-window:], y_pred[-window:])    
		else:
			acc = accuracy(y_real, y_pred)

	return y_real, y_pred, acc


#%% VALIDATION

if __name__  == "__main__":

	# LOAD IMAGES
	filepath1 = 'validation_images_car_cycle/ClassA/car_validation (1).jpg'
	image1_raw = utils.validation_load_B(filepath1, 256) # Load image
	image1_raw  = np.random.uniform(low = -1, high = 1, size = image1_raw.shape) + image1_raw # Add noise
	image1 = np.expand_dims(image1_raw, axis=0) # Add a dimension for batch
	
	filepath2 = 'validation_images_car_cycle/ClassA/car_validation (2).jpg'
	image2_raw = utils.validation_load_B(filepath2, 256) # Load image
	image2_raw  = np.random.uniform(low = -1, high = 1, size = image2_raw.shape) + image2_raw # Add noise
	image2 = np.expand_dims(image2_raw, axis=0) # Add a dimension for batch

	filepath3 = 'validation_images_car_cycle/ClassA/car_validation (3).jpg'
	image3_raw = utils.validation_load_B(filepath3, 256) # Load image
	image3_raw  = np.random.uniform(low = -1, high = 1, size = image3_raw.shape) + image3_raw # Add noise
	image3 = np.expand_dims(image3_raw, axis=0) # Add a dimension for batch
	
	filepath4 = 'validation_images_car_cycle/ClassA/car_validation (4).jpg'
	image4_raw = utils.validation_load_B(filepath4, 256) # Load image
	image4_raw  = np.random.uniform(low = -1, high = 1, size = image4_raw.shape) + image4_raw # Add noise
	image4 = np.expand_dims(image4_raw, axis=0) # Add a dimension for batch

	concat1 = tf.concat([image1, image2], 0)
	concat2 = tf.concat([image3, image4], 0)
	concat_all = tf.concat([image1, image2, image3, image4], 0)

	# PRINT IMAGES (IF SO)
	'''
	import matplotlib.pyplot as plt
	plt.imshow(image1_raw * 0.5 + 0.5) # getting the pixel values between [0, 1] to plot it.
	plt.show()
	plt.figure()
	plt.imshow(image2_raw * 0.5 + 0.5) # getting the pixel values between [0, 1] to plot it.
	plt.show()
	'''

	# INCEPTION SCORE
	t = time.time()
	is_score = get_inception_score(concat_all)
	print("IS = {:.2f}".format(is_score))
	dt_np = time.time() - t
	print("A avaliação do IS com Numpy levou {:.2f} s".format(dt_np))

	# FRECHET INCEPTION DISTANCE
	t = time.time()
	fid_score = get_frechet_inception_distance(concat1, concat2)
	print("FID = {:.2f}".format(fid_score))
	dt_np = time.time() - t
	print("A avaliação do FID com Numpy levou {:.2f} s".format(dt_np))

	# L1
	t = time.time()
	l1 = get_l1_distance(concat1, concat2)
	print("L1 = {:.2f}".format(l1))
	dt_np = time.time() - t
	print("A avaliação do L1 com TF levou {:.2f} s".format(dt_np))
