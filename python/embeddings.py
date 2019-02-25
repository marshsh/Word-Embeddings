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




def smh_embeddings_from_model( model, vocabulary ):
	# model.ldb[0][0].item

	model = smh.listdb_load(folder+prefix+'.models')

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



