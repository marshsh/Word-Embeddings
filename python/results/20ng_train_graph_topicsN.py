import python.train as train
import python.corpus as corpus
import python.arguments as arguments
import argparse


def main(args2):

	topic_N_s = [x for x in range(args2.start, args2.end, args2.step)]

	print " Training SMH reduced to  _topicN_  topics with the following values : \n {} \n ".format(str(topic_N_s))

	for topicN in topic_N_s:
		args = arguments.preMain(["-e", "smh", "-tN", str(topicN), "-tS", str(args2.tS), "-coo", str(args2.coo)], "--overlap", str(args2.ov))
		train.main(args)


def preMain():

	parser = argparse.ArgumentParser()

	parser.add_argument( "--start", "-s", "-start", default=1000, type=int, help="The first TopicN in list ")

	parser.add_argument( "--end", "-end", "-e", default=7000, type=int, help="The last TopicN in list ")

	parser.add_argument( "--step", "-step", "-st", default=1000, type=int, help="The Steps-length of TopicN in list ")

	parser.add_argument( "--coo", "-coo", default=arguments.COOCURRENCE_THRESHOLDS, type=float, help="COOCURRENCE_THRESHOLDS")

	parser.add_argument( "--tupleSize", "-tS", default=arguments.TUPLE_SIZE, type=float, help="TUPLE_SIZE")

	parser.add_argument( "--overlap", "-ov", default=arguments.OVERLAP, type=float, help="OVERLAP")


	args2 = parser.parse_args()

	print " \n \n Received this arguments :    {}".format(args2)

	return args2


if __name__ == "__main__":

	args2 = preMain()
	main(args2)
