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



def smh_20ng_get_embedding():
	filePrefix = os.path.join( 'data', '20newsgroups', '20newsgroups' )

	# We suppose '20newsgroups' dataSet has already been pre-processed with 
	# the 'prepare_db.sh' script.

	smh_get_model(filePrefix)
	smhVectors = smh_embeddings_from_model( filePrefix )

	return smhVectors




#################################################################################################
# SMH

def smh_get_model( filePrefix ):

    dir_path = os.getcwd()
    dir_smh = os.path.join( dir_path, filePrefix)



	corpus = smh.listdb_load(filePrefix + '.corpus')
	ifs = smh.listdb_load(filePrefix + '.ifs')
	discoverer = smh.SMHDiscoverer()
	models = discoverer.fit(ifs, expand = corpus)
	models.save(filePrefix + '.models')



# All preparations needed to use SMH, are done in the script prepare_db.sh
def smh_embeddings_from_model( filePrefix ):

	model = smh.listdb_load(filePrefix + '.models')
	vocabulary = smh.listdb_load(filePrefix + '.models')

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

	return smhVectors



