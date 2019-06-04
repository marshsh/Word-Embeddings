#!/bin/bash
#
# Script to run experiments.
#



coos='0.04 0.06 0.08 0.10'

for coo in $coos
do
	python ./python/train.py -e smh -tN 7000 -tS 3 -coo $coo --overlap 0.8
	echo '********************************************'
	echo Next experiments
	echo '********************************************'
done

for coo in $coos
do
	python ./python/train.py -e smh -tN 7000 -tS 2 -coo $coo --overlap 0.8
	echo '********************************************'
	echo Next experiments
	echo '********************************************'
done


echo All done





