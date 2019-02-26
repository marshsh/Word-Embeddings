import os
import smh



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
	# the SMH vectors have already been calculated and saved
	if os.path.exists(filePrefix + '.smh_vectors'):
		return loadDic(filePrefix + '.smh_vectors')

	# the vectors have not been calculated, but the topic distribution have been saved
	if os.path.exists(filePrefix + '.topicsRaw'):
		return smh_embeddings_from_model( filePrefix )


	# We calculate all from the documents' bags of words
	smh_get_model(filePrefix)
	smhVectors = smh_embeddings_from_model( filePrefix )
	return smhVectors





def contextSMH_get_embeddings( filePrefix, windowSize = 5 ):

	embeddings_dic = {}

	smhVectors = {}	
	contextVecBefore = {}
	contextVecAfter = {}

	# Load saved context vectors
	if os.path.exists(filePrefix + '.ctxtBefore' + '.' + str(windowSize)):
		contextVecBefore = loadDic(filePrefix + '.ctxtBefore' + '.' + str(windowSize))
		contextVecAfter = loadDic(filePrefix + '.ctxtAfter' + '.' + str(windowSize))
	else:
		# the SMH vectors have already been calculated and saved, but CTXT vectors haven't
		if os.path.exists(filePrefix + '.smh_vectors'):
			smhVectors = loadDic(filePrefix + '.smh_vectors')
		else :
			smhVectors = smh_get_embeddings( filePrefix )

		contextVecBefore, contextVecAfter = contextSMH(filePrefix, smhVectors, windowSize)


	# Concatenation of embeddings.
	for key in contextVecBefore.keys():
		embeddings_dic[key] = contextVecBefore[key] + smhVectors[key] + contextVecAfter[key]

	return embeddings_dic


def glove_and_context_embeddings(filePrefix, windowSize = 5 ):

	glove = gloveEmbbedingDic()
	context = contextSMH_get_embeddings(filePrefix, windowSize = 5 )

	embeddings_dic = {}

	for key in context.keys():
		embeddings_dic[key] = context[key] + glove[key]

	return embeddings_dic



#################################################################################################
# SMH Vectors

def smh_get_model( filePrefix ):

	corpus = smh.listdb_load(filePrefix + '.corpus')
	ifs = smh.listdb_load(filePrefix + '.ifs')
	discoverer = smh.SMHDiscoverer()
	models = discoverer.fit(ifs, expand = corpus)
	models.save(filePrefix + '.topicsRaw')



# All preparations needed to use SMH, are done in the script prepare_db.sh
def smh_embeddings_from_model( filePrefix ):

	model = smh.listdb_load(filePrefix + '.topicsRaw')
	vocabulary = smh.listdb_load(filePrefix + '.topicsRaw')

	# vocabulary = getVocaularyDic()  # Falta Aquiiiii

	smhVectors = {}

	for topicId in range(model.ldb.size):
		for itemInList in range( model.ldb[topicId].size ):

			token = model.ldb[topicId][itemInList].item
			freq = model.ldb[topicId][itemInList].freq

			word = vocabulary[token]

			if word not in smhVectors:
				smhVectors[ word ] = [ 0 for n in range(model.ldb.size)]			


			smhVectors[word][topicId] = freq

	saveDic( filePrefix + '.smh_vectors' )

	return smhVectors


################################################################################
# SMH with context vectors


def contextSMH(filePrefix, smhVectors, windowSize):

	documentsFile = filePrefix + '.ref'

	contextVecBefore = {}
	contextVecAfter = {}

	with open(documentsFile, 'r') as f:
		for line in f.readlines():
			line = line.split(' ')
			length = len(line)

			for i, word in enumerate(line):
				for h in range(1,windowSize+1):
					if i+h < length :
						contextVecAfter[word] = contextVecAfter[word] + smhVectors[line[ i+h ]]
					if i-h > -1 :
						contextVecBefore[word] = contextVecBefore[word] + smhVectors[line[ i-h ]]

	saveDic(contextVecBefore, filePrefix + '.ctxtBefore' + '.' + str(windowSize) )
	saveDic(contextVecAfter, filePrefix + '.ctxtAfter' + '.' + str(windowSize) )

	return contextVecBefore, contextVecAfter						















#####################################################################################
# Usefull functions


def saveDic(dic,fileName):
	with open(fileName, 'w') as f:
		for key, val in dic.items():
			f.write( [key, val] )


def loadDic(fileName):
	dic = {}
	with open(fileName, 'r') as f:
		for line in f.readlines():
			key = line[0]
			val = line[1:]
			dic[key] = val
	return dic




