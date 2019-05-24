import argparse
import os

from gensim.models.ldamodel import LdaModel

import corpus
import tools
import arguments as a




def lda_embeddings(corpusName, num_topics=None, epochsN=None, reCalculate=False):


	nameDrop = "./data/{}/{}.lda_emb_{}".format(corpusName, corpusName, getLDAextension(num_topics,epochsN))

	if os.path.exists(nameDrop) and (not reCalculate) :
		dic = tools.loadPickle(nameDrop)
		return dic


	corp = corpus.getCorpus(corpusName)

	it = corp.LDA_stream_x_train()

	print "Training LDA topics model with: _{}, in _{}_ epochs.".format(num_topics,epochsN)



	lda = LdaModel(it,num_topics=num_topics, passes=epochsN)
	num_topics_new = lda.num_topics

	dic = {}
	notIn = 0

	print "Creating dictionary of LDA's trained wordembeddings."
	for i in range(a.MAX_NUM_WORDS):
	    try :
	    	ldaVec = lda.get_term_topics(i)
	    	vec = ldaToVector(ldaVec, num_topics_new)
	    	dic[i] = vec
	    except KeyError:
	    	notIn += 1

	print "Saving Dictionary"
	tools.dumpPickle(nameDrop, dic)
	print "Dictionary Saved"

	return dic




def ldaToVector(ldaVec, num_topics):
	vec = [0 for x in range(num_topics)]
	for pair in ldaVec:
		i, num = pair
		vec[i] = num
	return vec






def getLDAextension(num_topics,epochsN):
	extension = '[NumTopics_{}][Epochs_{}]'.format(num_topics,epochsN)
	return extension



# def main():


# 	corp = corpus.getCorpus(args.corpus, args.epochsN)


# if __name__ == '__main__':

# 	parser = argparse.ArgumentParser()

# 	parser.add_argument("--corpus", "-c", 
# 						choices=[ '20NG', '20ng', 'r', 'reuters', 'w', 'wiki', 'wikipedia'],
# 						help="Corpus to be used", 
# 						default='20ng')
# 	parser.add_argument("--epochsN","-ep", type=int,
# 						default=5 )

# 	args = parser.parse_args()



# 	# Unifiying corpus names
# 	if args.corpus in ['20NG','20ng']:
# 		args.corpus = '20newsgroups'
# 	if args.corpus in ['r', 'reuters']:
# 		args.corpus = 'reuters'
# 	if args.corpus in ['w', 'wiki', 'wikipedia']:
# 		args.corpus = 'wikipedia'

# 	# Adjusting epochsN
# 	if args.epochsN < 1:
# 		args.epochsN = 5

# 	main(args)




