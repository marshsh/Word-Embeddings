


import smh
import argparse
import os



def writeModel():

	corpus = smh.listdb_load(folder+prefix+'.corpus')
	ifs = smh.listdb_load(folder+prefix+'.ifs')
	discoverer = smh.SMHDiscoverer()
	models = discoverer.fit(ifs, expand = corpus)

	models.save(folder+prefix+'.models')




def getVocaularyDic():
	vocabulary = {}
	with open('knowceans-ilda/nips/nips.vocab', 'r') as f:
		content = f.readlines()
		for line in content:
	        	tokens = line.split(' = ')
	        	vocabulary[int(tokens[1])] = tokens[0]

	return vocabulary











# COPIADO a Embeddings (MODIFICAR ALLA TAMBIEN)

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





def context_SMH_embedding(smhVectors, vocabulary, windowSize):

	sizeSMH = len( next(iter(smhVectors.values())) )
	contextVectors = { key : [ [ 0 for i in range(sizeSMH) ]    for x in range(windowSize) ]   for key in list(smhVectors.keys()) }


# Falta Terminar de Escribir.





if __name__ == '__main__':


	parser = argparse.ArgumentParser()
    parser.add_argument( "folder",
    					help = "Folder where SMH saves ifs, models, etc ... ")
    parser.add_argument( "prefix",
    					help = "prefix to files")
    parser.add_argument( "corpus",
    					help = " Precompiled frequency of words in each file of corpus. (Bags Of Words) ")


    args = parser.parse_args()






