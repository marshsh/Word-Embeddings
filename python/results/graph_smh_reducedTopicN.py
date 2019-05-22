import matplotlib.pyplot as plt
import numpy as np
import python.results.values as v
import python.tools as tools
import os



fileName = v.fileName



# with open(fileName, 'r') as file:

def jbkabvk:
	''' Aqu creamos la lista de los archivos Pickle History ... donde se 
	encuentra la historia de entrenamiento de los modelos que queremos '''


for file in skglvl:
	a = tools.loadPickle('history/20newsgroups_gensim_conv+lstm_[32-128]___5-11--0:12')
	ac = a['acc'][-10:]
	train_a = sum(ac)/len(ac)
	vac = a['val_acc'][-10:]
	train_a = sum(vac)/len(vac)


# Crear grafica

