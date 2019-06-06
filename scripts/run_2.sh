#!/bin/bash
#
# Script to run experiments.
#


# Faltan (tS 2, coo 0.06, overlap 0.8, minClust 5)
# Faltan (tS 2, coo 0.04, overlap 0.8, minClust 5)


coos='0.02 0.04 0.06 0.08 0.10'

for coo in $coos
do
	python ./python/train.py --reCalculate -e smh -tN 7000 -tS 2 -coo $coo --overlap 0.8 --nameBoard 'yaa'

	echo
	echo
	echo
	echo
	echo '********************************************'
	echo Next experiments
	echo '********************************************'
	echo
	echo
	echo
	echo
done


coos='0.04 0.06 0.08 0.10'

for coo in $coos
do
	python ./python/train.py --reCalculate -e smh -tN 7000 -tS 3 -coo $coo --overlap 0.8 --nameBoard 'yaa'
	echo
	echo
	echo
	echo
	echo
	echo '********************************************'
	echo Next experiments
	echo '********************************************'
	echo
	echo
	echo
	echo
	echo
done


echo All done





