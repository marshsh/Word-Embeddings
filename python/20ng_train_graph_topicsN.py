import python.train as train
import python.corpus as corpus
import python.arguments as arguments
import python.results.values as v


x_values = v.x_values


for topicN in x_values:
	args = arguments.preMain(["-e", "smh", "-tN", str(topicN), "-tS", "2", "-coo", "0.02"])
	train.main(args)

for topicN in x_values:
	args = arguments.preMain(["-e", "smh", "-tN", str(topicN), "-tS", "2", "-coo", "0.03"])
	train.main(args)

for topicN in x_values:
	args = arguments.preMain(["-e", "smh", "-tN", str(topicN), "-tS", "3", "-coo", "0.02"])
	train.main(args)

for topicN in x_values:
	args = arguments.preMain(["-e", "smh", "-tN", str(topicN), "-tS", "3", "-coo", "0.03"])
	train.main(args)
