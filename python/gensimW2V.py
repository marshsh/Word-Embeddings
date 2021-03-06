
import argparse
from gensim.models import Word2Vec
import os

import corpus
import tools
import arguments as a


def gensimW2V_embeddings(corpusName, epochsN=5, reCalculate=False):

	nameDrop = "./data/{}/{}.gensim_emb_{}".format(corpusName, corpusName, getGensimExtension(epochsN))

	if os.path.exists(nameDrop) and (not reCalculate) :
		dic = tools.loadPickle(nameDrop)
		return dic


	corp = corpus.getCorpus(corpusName)

	it = corp.w2v_iterator()

	print "Training gensim word embedings of dimension: {}, in {} epochs.".format(a.EMBEDDING_DIM,epochsN)
	model = Word2Vec( it, size=a.EMBEDDING_DIM, min_count=5)
	model.build_vocab(it)
	model.train(it, total_examples=model.corpus_count, epochs=epochsN, compute_loss=True)

	dic = {}
	notIn = 0

	print "Creating dictionary of gensim's trained wordembeddings."
	for i in range(a.MAX_NUM_WORDS):
	    try :
	    	vec = model.wv[str(i)]
	    	dic[i] = vec
	    except KeyError:
	    	notIn += 1

	print "Saving Dictionary"
	tools.dumpPickle(nameDrop, dic)
	print "Dictionary Saved"

	return dic



def getGensimExtension(epochsN):
	extension = '[embSize_{}][epochs_{}]'.format(a.EMBEDDING_DIM, epochsN)
	return extension


def main():


	corp = corpus.getCorpus(args.corpus, args.epochsN)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--corpus", "-c", 
						choices=[ '20NG', '20ng', 'r', 'reuters', 'w', 'wiki', 'wikipedia'],
						help="Corpus to be used", 
						default='20ng')
	parser.add_argument("--epochsN","-ep", type=int,
						default=5 )

	args = parser.parse_args()



	# Unifiying corpus names
	if args.corpus in ['20NG','20ng']:
		args.corpus = '20newsgroups'
	if args.corpus in ['r', 'reuters']:
		args.corpus = 'reuters'
	if args.corpus in ['w', 'wiki', 'wikipedia']:
		args.corpus = 'wikipedia'

	# Adjusting epochsN
	if args.epochsN < 1:
		args.epochsN = 5

	main(args)

