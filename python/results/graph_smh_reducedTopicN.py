import matplotlib.pyplot as plt
import numpy as np
import python.results.values as v
import python.tools as tools
import python.argumets as a
import python.embeddings as emb

import os


# Parametros como estan ahorita.

TUPLE_SIZE = 2 # 3     This is r.
COOCURRENCE_THRESHOLDS = 0.02 # 0.03
OVERLAP = 0.9
MIN_CLUSTER_SIZE = 5 # 10

TOP_TOPIC_WORDS = 10




fileName = v.fileName




x_graph = [] # Los valores de x de la grafica (los topicN de los que si tenemos history)
y_graph_train = [] # Los valores de y de la grafica 
y_graph_val = [] # Los valores de y de la grafica 
histories_list = [] # los nombres de los PickleFile que contienen la historia de entrenamiento


# Recorremos el directorio *history* para ver cuales modelos hemos entrenado
dirsHist = os.listdir( 'history' )


# Para cada topicN_ encontramos una historia de entrenmiento y los agregamos a las listas (si hay hist.)
for topicN_ in v.x_values:
	extension = getSMHextension(embType='', tupSize=TUPLE_SIZE, coo=COOCURRENCE_THRESHOLDS, 
		overlap=OVERLAP, minClustS=MIN_CLUSTER_SIZE, topicN=topicN_)

	listDirs = filter(lambda x : extension in x, dirsHist )

	if listDirs :
		historyDir = listDirs[-1]
		x_graph.append(topicN_)
		histories_list.append(historyDir)


# Para cada history de entrenamiento, calculamos el accuracy promedio de las ultimas 5 epocas, 
# y agregamos el valor a y_graph
for file in histories_list:
	a = tools.loadPickle(file)

	ac = a['acc'][-5:]
	train_acc = sum(ac)/len(ac)
	y_graph_train.append(train_acc)

	vac = a['val_acc'][-5:]
	val_acc = sum(vac)/len(vac)
	y_graph_val.append(val_acc)


# Ya tenemos el acc y val_acc de los entrenamientos

# Crear grafica







