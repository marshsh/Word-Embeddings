import argparse
import embeddings
import os

# Global Variables

EPOCHS = 100


MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.18

WINDOW_SIZE = 5


TUPLE_SIZE = 2
COOCURRENCE_THRESHOLS = 0.02
OVERLAP = 0.9


def preMain(aaaargs=[]):
	"""
	With this method, you get the 'args' object you would've gotten had you run 
	this script directly with the corresponding options.

	USAGE:
	In python environment:

	import train
	args = train.preMain(["-e", "w2v", "-c", "20ng"])
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument("--layout", "-ly", "-l", 
						choices=['words','docs'], 
						default = 'words',
						help="Type of classification embeddings we're creating.")

	parser.add_argument("--embedding_type", "-et", "-e", 
						choices=['smh', 'oneH', 'w2v', 'glove', 'contextVec', 'topicAvg', 
						'w2v+smh', 'w2v+contextVec', 'glove+contextVec', 'w2v+topicAvg', 
						'w2v+context', 'w2vGensim'   ], 
						help="Type of word representation used to train the model.")

	parser.add_argument("--corpus", "-c", 
						choices=[ '20NG', '20ng', 'r', 'reuters', 'w', 'wiki', 'wikipedia'],
						help="Corpus to be used", 
						default='20ng')

	parser.add_argument("--kerasModel", "-km", "-model", "-keras", 
						choices=['conv', 'lstm', 'conv+lstm'],
						default='conv+lstm',
						help="Architecture of the neural network used to classify texts")

	


	parser.add_argument("--convFilters", "-convF", type=int, default=32, help="Number of Conv1D Filters used in conv1D Keras Model")

	parser.add_argument("--lstmNeurons", "-lstmN", "-lstm", type=int, default=128, help="Number of neurons in lstm layer of Keras Model")

	parser.add_argument("--size", type=int)


# SMH Parameters
	parser.add_argument("--tupleS", type=int)

	parser.add_argument("--coo_threshold", "-coo", type=float)

	parser.add_argument("--overlap", "-cv", type=float)




	parser.add_argument("--nameBoard", type=str)

	parser.add_argument("--nameCorpus", type=str, default='')

	parser.add_argument("--reCalculate", 
						help="re-calculate chosen word-vector embeddings", 
						action="store_true")
	
	parser.add_argument("--logNormal", 
						help="utilize log-Normalization in smh word-vector embeddings", 
						action="store_true")

	parser.add_argument("--restore", 
						help="restore Keras model from latest training moment", 
						action="store_true")

	# Parsing Arguments
	if aaaargs:
		args = parser.parse_args(aaaargs)
	else :
		args = parser.parse_args()
	print " \n Training ", args.corpus, "with ", args.embedding_type, " embbedings"


	# Adding logNormal label 
	if args.logNormal:
		args.embedding_type += "_logN"
		print "Using _logNormal smh embeddings."

	# Unifiying corpus names
	if args.corpus in ['20NG','20ng']:
		args.corpus = '20newsgroups'
	if args.corpus in ['r', 'reuters']:
		args.corpus = 'reuters'
	if args.corpus in ['w', 'wiki', 'wikipedia']:
		args.corpus = 'wikipedia'


# SMH Parameters
	if args.tupleS :
		global TUPLE_SIZE
		if args.tupleS < 2:
			TUPLE_SIZE = 2
		else :
			TUPLE_SIZE = args.tupleS
		
	if args.coo_threshold :
		global COOCURRENCE_THRESHOLS
		COOCURRENCE_THRESHOLS = args.coo_threshold

	if args.overlap :
		global OVERLAP
		OVERLAP = args.overlap



# PREFIX fix
	# Adding file-prefix to have a well organized way of saving pre-calculated embeddings.
	filePrefix = 'data/'
	if args.corpus in ['20NG', '20ng', '20newsgroups']:
		filePrefix = os.path.join(filePrefix, '20newsgroups', '20newsgroups')
	elif args.corpus in ['r', 'reuters']:
		filePrefix = os.path.join(filePrefix, 'reuters', 'reuters')
	elif args.corpus in ['w', 'wiki', 'wikipedia']:
		filePrefix = os.path.join(filePrefix, 'wikipedia', 'wiki')
	else :
		print " \n Couldn't find corresponding filePrefix \n"

	if args.logNormal:
		filePrefix += '_logNorm'  #If you change this string '_logNorm' you have to change it in embeddings.smh_get_model() too

	args.filePrefix = filePrefix

	print "\n \n \n \n" + filePrefix + "\n \n"


# NAME fix
	# Initiating Extra Name
	if not args.nameBoard:
		args.nameBoard = ''
	else :
		args.nameBoard =  '(' + args.nameBoard + ')'


	# Adding SMH minTuppleSize and coocurringThreshold
	lista = ['smh', 'context']
	if bool(sum(map( lambda x: x in args.embedding_type, lista))):
		smhName = embeddings.getSMHextension()
		args.nameBoard = smhName + args.nameBoard

	# FINAL NAME
	args.nameBoard = "{}_{}_{}_[{}-{}]_{}".format(args.corpus, args.embedding_type, 
		args.kerasModel, args.convFilters, args.lstmNeurons, args.nameBoard)



	if args.size == None:
		args.size = WINDOW_SIZE

	return args


