import os
import time
import numpy as np

from scipy.linalg import sqrtm
from sklearn.metrics import accuracy_score as accuracy

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

def evaluate_metrics_cyclegan(sample_ds_A, sample_ds_B, generator_g, generator_f, discriminator_x, discriminator_y,
							  evaluate_is, evaluate_fid, evaluate_acc, verbose = False):
	t1 = time.time()
	inception_score_A = []
	inception_score_B = []
	frechet_inception_distance_A = []
	frechet_inception_distance_B = []
	accuracies_A = []
	accuracies_B = []
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

			# Cálculos da Acurácia dos discriminadores
			if evaluate_acc:
				acc_A = evaluate_disc_accuracy_cyclegan(discriminator_x, input_A, fake_A)
				acc_B = evaluate_disc_accuracy_cyclegan(discriminator_y, input_B, fake_B)
				accuracies_A.append(acc_A)
				accuracies_B.append(acc_B)
				if verbose: 
					print("ACC (A) = {:.2f}".format(acc_A))
					print("ACC (B) = {:.2f}".format(acc_B))

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
	if evaluate_acc:
		acc_avg_A, acc_std_A = np.mean(accuracies_A), np.std(accuracies_A)
		results['acc_avg_A'] = acc_avg_A
		results['acc_std_A'] = acc_std_A
		acc_avg_B, acc_std_B = np.mean(accuracies_B), np.std(accuracies_B)
		results['acc_avg_B'] = acc_avg_B
		results['acc_std_B'] = acc_std_B

	# Reporta o resultado
	if verbose:
		if evaluate_is:
			print("Inception Score A:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(is_avg_A, is_std_A))
			print("Inception Score B:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(is_avg_B, is_std_B))
		if evaluate_fid:
			print("Fréchet Inception Distance A:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(fid_avg_A, fid_std_A))
			print("Fréchet Inception Distance B:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(fid_avg_B, fid_std_B))
		if evaluate_acc:
			print("Discriminator Accuracy A:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(acc_avg_A, acc_std_A))
			print("Discriminator Accuracy B:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(acc_avg_B, acc_std_B))

	dt = time.time() - t1
	results['eval_time'] = dt

	return results

def evaluate_metrics_pix2pix(sample_ds, generator, discriminator, evaluate_is, evaluate_fid, evaluate_l1, evaluate_acc, verbose = False):
	t1 = time.time()
	inception_score = []
	frechet_inception_distance = []
	l1_distance = []
	accuracies = []
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

			# Acurácia do Discriminador
			if evaluate_acc:
				acc = evaluate_disc_accuracy_pix2pix(discriminator, fake, target)
				accuracies.append(acc)
				if verbose: 
					print("ACC = {:.2f}".format(acc))

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
	if evaluate_acc:
		acc_avg, acc_std = np.mean(accuracies), np.std(accuracies)
		results['acc_avg'] = acc_avg
		results['acc_std'] = acc_std

	# Reporta o resultado
	if verbose:
		if evaluate_is:
			print("Inception Score:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(is_avg, is_std))
		if evaluate_fid:
			print("Fréchet Inception Distance:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(fid_avg, fid_std))
		if evaluate_l1:
			print("L1 Distance:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(l1_avg, l1_std))
		if evaluate_acc:
			print("Discriminator Accuracy:\nMédia: {:.2f}\nDesv Pad: {:.2f}\n".format(acc_avg, acc_std))

	dt = time.time() - t1
	results['eval_time'] = dt

	return results


#%% FUNÇÕES DE CÁLCULO DAS MÉTRICAS

# Inception Score
def get_inception_score(image):
	
	'''
	Calcula o Inception Score (IS) para uma única imagem. Baseado em:
    https://machinelearningmastery.com/how-to-implement-the-inception-score-from-scratch-for-evaluating-generated-images/
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
	Calcula o Fréchet Inception Distance (FID) entre duas imagens. Baseado em:
    https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
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

	'''
	Calcula a distância L1 entre duas imagens
	'''

	# Calcula a L1 distance entre as duas imagens
	l1_dist = tf.reduce_mean(tf.abs(image1 - image2))

	return l1_dist

# Acurácia do Discriminador - Pix2Pix
def evaluate_disc_accuracy_pix2pix(disc, fake_image, target):
    
	'''
	Calcula a acurácia do discriminador
	'''

	# Realiza as discriminações
	disc_real = disc([target, target])
	disc_fake = disc([fake_image, target])

	# Para o caso de ser um discriminador PatchGAN, tira a média
	disc_real = np.mean(disc_real)
	disc_fake = np.mean(disc_fake)

	# Aplica o threshold
	disc_real = 1 if disc_real > 0.5 else 0
	disc_fake = 1 if disc_fake > 0.5 else 0

	# Prepara as listas para poder realizar o cálculo da acurácia
	y_real = []
	y_pred = []

	# Acrescenta a observação real como y_real = 1
	y_real.append(1)
	y_pred.append(disc_real)

	# Acrescenta a observação fake como y_real = 0
	y_real.append(0)
	y_pred.append(disc_fake)

	# Calcula a acurácia
	acc = accuracy(y_real, y_pred)

	return acc

# Acurácia do Discriminador - CycleGAN
def evaluate_disc_accuracy_cyclegan(disc, real_image, fake_image):
    
	'''
	Calcula a acurácia do discriminador
	'''

	# Realiza as discriminações
	disc_real = disc(real_image)
	disc_fake = disc(fake_image)

	# Para o caso de ser um discriminador PatchGAN, tira a média
	disc_real = np.mean(disc_real)
	disc_fake = np.mean(disc_fake)

	# Aplica o threshold
	disc_real = 1 if disc_real > 0.5 else 0
	disc_fake = 1 if disc_fake > 0.5 else 0

	# Prepara as listas para poder realizar o cálculo da acurácia
	y_real = []
	y_pred = []

	# Acrescenta a observação real como y_real = 1
	y_real.append(1)
	y_pred.append(disc_real)

	# Acrescenta a observação fake como y_real = 0
	y_real.append(0)
	y_pred.append(disc_fake)

	# Calcula a acurácia
	acc = accuracy(y_real, y_pred)

	return acc