#!/bin/bash
#
# Script to run experiments.
#


print_space () {
	echo
	echo
	echo
	echo
	echo '********************************************'
	echo Next experiment
	echo '********************************************'
	echo
	echo
	echo
	echo

}


print_space


# coos='0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6'
# coos='0.4 0.5 0.6'

# for coo in $coos
# do
# 	python ./python/train.py -e smh -tN 7000 -tS 4 -coo $coo --overlap 0.8 --nameBoard 'yaya'

# 	print_space

# done

python ./python/train.py -e gensim --embSize 300 -ep 5 -km lstm --reCalculate
print_space
python ./python/train.py -e gensim --embSize 300 -ep 10 -km lstm
print_space
python ./python/train.py -e gensim --embSize 300 -ep 50 -km lstm
print_space



python ./python/train.py -e smh -tN 7000 -tS 2 -coo 0.3 --overlap 0.8 --nameBoard 'ya_StopWords'
print_space
python ./python/train.py -e smh -tN 7000 -tS 3 -coo 0.35 --overlap 0.8 --nameBoard 'ya_StopWords'
print_space
python ./python/train.py -e smh -tN 7000 -tS 4 -coo 0.45 --overlap 0.8 --nameBoard 'ya_StopWords'
print_space
python ./python/train.py -e smh -tN 7000 -tS 5 -coo 0.45 --overlap 0.8 --nameBoard 'ya_StopWords'
print_space
python ./python/train.py -e smh -tN 7000 -tS 5 -coo 0.5 --overlap 0.8 --nameBoard 'ya_StopWords'
print_space
python ./python/train.py -e smh -tN 7000 -tS 5 -coo 0.55 --overlap 0.8 --nameBoard 'ya_StopWords'
print_space


echo
echo
echo All done





