import os
import smh
import numpy as np
import pickle
from collections import Iterable

from discovery.topics import load_vocabulary, save_topics, save_time, get_models_docfreq, sort_topics, listdb_to_topics


def gloveEmbbedingDic():
    """
    Returns dictionary with pre-trained word embbedings.
    glove.6B.2 must be downloaded to "pwd/3rdParty/glove.6B.2/glove.6B.100d.txt"
    before running this script.
    """

    # dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.getcwd()
    GLOVE_DIR = os.path.join( dir_path, "3rdParty", "glove.6B.2")

    embeddings_dic = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_dic[word] = coefs

    return embeddings_dic

	
def smh_get_embeddings( filePrefix ):
	print '*** smh_get_embeddings ***'
	# the SMH vectors have already been calculated and saved
	if os.path.exists(filePrefix + '.smh_vectors'):
		return loadPickle(filePrefix + '.smh_vectors')

	# the vectors have not been calculated, but the topic distribution have been saved
	if os.path.exists(filePrefix + '.topicsRaw'):
		return smh_embeddings_from_model( filePrefix )


	# We calculate all from the documents' bags of words
	smh_get_model(filePrefix)
	smhVectors = smh_embeddings_from_model( filePrefix )

	dumpPickle( filePrefix + '.smh_vectors', smhVectors )

	return smhVectors





def contextSMH_get_embeddings( filePrefix, windowSize = 5 ):

	if os.path.exists(filePrefix + '.context' + '.' + str(windowSize)):
		contextVec = loadPickle(filePrefix + '.context' + '.' + str(windowSize))
		return contextVec




	# Load saved context vectors
	if os.path.exists(filePrefix + '.ctxtBefore' + '.' + str(windowSize)):
		print 'Loading contextVecBefore and ... \n'
		contextVecBefore = loadPickle(filePrefix + '.ctxtBefore' + '.' + str(windowSize))
		contextVecAfter = loadPickle(filePrefix + '.ctxtAfter' + '.' + str(windowSize))
		# print contextVecBefore.keys()
	else:
		# the SMH vectors have already been calculated and saved, but CTXT vectors haven't
		if os.path.exists(filePrefix + '.smh_vectors'):
			smhVectors = loadPickle(filePrefix + '.smh_vectors')
		else :
			print 'Loading smhVectors \n'
			smhVectors = smh_get_embeddings( filePrefix )

		print 'Calculating contextVecBefore \n'
		contextVecBefore, contextVecAfter = contextSMH(filePrefix, smhVectors, windowSize)

		dumpPickle(filePrefix + '.ctxtBefore' + '.' + str(windowSize), contextVecBefore )
		dumpPickle(filePrefix + '.ctxtAfter' + '.' + str(windowSize), contextVecAfter )


	# print ' \n Concatenation of embeddings.'
	# for key in contextVecBefore.keys():
	# 	embeddings_dic[key] =  contextVecBefore[key] + contextVecAfter[key]
	# print 'Embeddings concatenated. \n'

	print 'Adding ContextAfter and ContextBefore into new dictionary'

	embeddings_dic = {}


	print 'Length of contextVecBefore.keys() : ', len(contextVecBefore.keys()) 
	sizeVectors = len(contextVecBefore[contextVecBefore.keys()[0]])

	for key in contextVecBefore.keys():
		embeddings_dic[key] =  [ contextVecBefore[key][x] + contextVecAfter[key][x] for x in range(sizeVectors)]
	print 'Embeddings concatenated. \n'


	dumpPickle(filePrefix + '.context' + '.' + str(windowSize), embeddings_dic)


	return embeddings_dic


def glove_and_context_embeddings(filePrefix, windowSize = 5 ):

	if os.path.exists(filePrefix + '.glove_and_context'):
		return loadPickle(filePrefix + '.glove_and_context')


	glove = gloveEmbbedingDic()
	context = contextSMH_get_embeddings(filePrefix, windowSize = 5 )

	embeddings_dic = {}
	sizeGlove = len(glove[glove.keys()[0]])

	for key in context.keys():
		extraV = glove.get(key)

		if not isinstance(extraV, Iterable):
			extraV = [ 0 for x in range(sizeGlove) ]

		embeddings_dic[key] = np.concatenate([context[key] , extraV])


	dumpPickle( filePrefix + '.glove_and_context', embeddings_dic )

	return embeddings_dic



#################################################################################################
# SMH Vectors

def smh_get_model( filePrefix ):
	print '*** smh_get_model ***'

	corpus = smh.listdb_load(filePrefix + '.corpus')
	ifs = smh.listdb_load(filePrefix + '.ifs')
	print 'Loaded .ref and .ifs'
	discoverer = smh.SMHDiscoverer()
	print 'Fitting SMH Discoverer'
	models = discoverer.fit(ifs, expand = corpus)
	models.save(filePrefix + '.topicsRaw')
	print "SMH Model saved (a ldb with lists of topics' tokens)"


# All preparations needed to use SMH, are done in the script prepare_db.sh
def smh_embeddings_from_model( filePrefix ):

	model = smh.listdb_load(filePrefix + '.topicsRaw')
	vocpath = filePrefix + '.vocab'
	print "Loading vocabulary from", vocpath
	vocabulary, docfreq = load_vocabulary(vocpath)



	smhVectors = {}

	for topicId in range(model.ldb.size):
		for itemInList in range( model.ldb[topicId].size ):

			token = model.ldb[topicId][itemInList].item
			freq = model.ldb[topicId][itemInList].freq

			word = vocabulary[token]

			if word not in smhVectors:
				smhVectors[ word ] = [ 0 for n in range(model.ldb.size)]			


			smhVectors[word][topicId] = freq

	# dumpPickle( filePrefix + '.smh_vectors', smhVectors ) # Already saving in calling method

	return smhVectors


################################################################################
# SMH with context vectors


def contextSMH(filePrefix, smhVectors, windowSize):

	documentsFile = filePrefix + '.ref'

	contextVecBefore = {}
	contextVecAfter = {}

	sizeVectors = len(smhVectors[smhVectors.keys()[0]])

	with open(documentsFile, 'r') as f:
		for line in f.readlines():
			line = line.split(' ')
			length = len(line)

			for i, word in enumerate(line):

				if smhVectors.get(word) != None:
					if contextVecAfter.get(word) == None :
						contextVecAfter[word] = smhVectors.get(word)
						contextVecBefore[word] = smhVectors.get(word)
				
				for h in range(1,windowSize+1):
					if i+h < length :
						if smhVectors.get(line[ i+h ]) != None:
							if smhVectors.get(word) != None:
								if contextVecAfter.get(word) != None:
									contextVecAfter[word] = [ contextVecAfter[word][x] + smhVectors.get(line[ i+h ])[x] for x in range(sizeVectors)  ]
					if i-h > -1 :
						if smhVectors.get(line[ i-h ]) != None:
							if smhVectors.get(word) != None:
								if contextVecBefore.get(word) != None:
									contextVecBefore[word] = [ contextVecBefore[word][x] + smhVectors[line[ i-h ]][x] for x in range(sizeVectors)  ]
				
	# dumpPickle(contextVecBefore, filePrefix + '.ctxtBefore' + '.' + str(windowSize) )
	# dumpPickle(contextVecAfter, filePrefix + '.ctxtAfter' + '.' + str(windowSize) )

	return contextVecBefore, contextVecAfter						















#####################################################################################
# Usefull functions


# def saveDic(dic,fileName):
# 	with open(fileName, 'w') as f:
# 		for key, val in dic.items():
# 			f.write( [key, val] )


# def loadDic(fileName):
# 	dic = {}
# 	with open(fileName, 'r') as f:
# 		for line in f.readlines():
# 			key = line[0]
# 			val = line[1:]
# 			dic[key] = val
# 	return dic



def dumpPickle(fileName, dic):
	pickle_out = open(fileName,"w")
	pickle.dump(dic, pickle_out)
	pickle_out.close()

def loadPickle(fileName):
	print 'Loading Pickle :  ' + fileName
	pickle_in = open(fileName,"r")
	dic = pickle.load(pickle_in)
	print 'Loading completed ... \n'
	return dic






	# Traceback (most recent call last):
	#   File "python/train.py", line 189, in <module>
	#     main()
	#   File "python/train.py", line 115, in main
	#     embedding_layer = getEmbeddingLayer(args.embedding_type, corpusA, MAX_NUM_WORDS, EMBEDDING_DIM)
	#   File "python/train.py", line 65, in getEmbeddingLayer
	#     embeddings_dic = embeddings.contextSMH_get_embeddings( args.filePrefix, args.size )
	#   File "/home/mariana/Sheeeeeik-Code/Word-Embbedings/Word-Embeddings/python/embeddings.py", line 83, in contextSMH_get_embeddings
	#     dumpPickle(contextVecBefore, filePrefix + '.ctxtBefore' + '.' + str(windowSize) )
	#   File "/home/mariana/Sheeeeeik-Code/Word-Embbedings/Word-Embeddings/python/embeddings.py", line 227, in dumpPickle
	#     pickle_out = open(fileName,"wb")
	# TypeError: coercing to Unicode: need string or buffer, dict found
