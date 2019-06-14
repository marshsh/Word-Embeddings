#!/bin/bash
#
# Script to run experiments.
#


# Faltan (tS 2, coo 0.06, overlap 0.8, minClust 5)
# Faltan (tS 2, coo 0.04, overlap 0.8, minClust 5)


coos='0.4 0.5 0.6 0.7'

for coo in $coos
do
	python ./python/train.py --reCalculate -e smh -tN 7000 -tS 2 -coo $coo --overlap 0.8 --nameBoard 'yaya'

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


python ./python/train.py -e glove -km conv -sL 300 #--nameBoard 'yaya'

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


coos='0.1 0.2 0.3 0.4 0.5 0.6 0.7'

for coo in $coos
do
	python ./python/train.py --reCalculate -e smh -tN 7000 -tS 3 -coo $coo --overlap 0.8 --nameBoard 'yaya'
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





