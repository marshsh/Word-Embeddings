
import arguments as a
import train



COO = [0.04, 0.06, 0.08, 0.10]


for coo in COO:

	args = a.preMain([ "-e", "smh", "-tN", "7000", "-tS", "3", "-coo", str(coo), "--overlap", "0.8"])

	corpusA = train.getCorpus(args)

	embedding_layer = train.getEmbeddingLayer(args, args.embedding_type, corpusA, a.MAX_NUM_WORDS, a.EMBEDDING_DIM)











