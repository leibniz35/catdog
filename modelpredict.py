from fastai.vision import *
from urllib.request import urlretrieve
from os import path
from random import choice

class ModelPredict():
	def __init__(self, filename):
		self.filename = filename

	def download_model(self):
		if path.exists('export.pkl') == False:
			url = 'https://drive.google.com/uc?export=download&id=1e-CSvNRED7__dwWsvSZ95ydsE74vSz-3'
			filename = 'export.pkl'
			urlretrieve(url,filename)

	def predict(self):
		self.download_model()
		learn = load_learner('')
		img = open_image(self.filename)
		pred_class , pred_idx, outputs = learn.predict(img)
		return str(pred_class)

if __name__=='__main__':
	m = ModelPredict('kedi.jpg').predict()
	print(m)
