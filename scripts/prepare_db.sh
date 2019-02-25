#!/bin/bash
#
# Script to
#
#
ROOTPATH=`pwd`
THIRDPARTYPATH=$ROOTPATH/3rdParty
DATAPATH=$ROOTPATH/data
mkdir -p $DATAPATH
WIKIDUMPEN="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"

GLOVENAME=$THIRDPARTYPATH/glove.6B.zip



TWENTYNG=false
REUTERS=false
WIKIPEDIA=false
WIKI2TEXT=false

SMH=false
CTXT=false
W2V=false
GLOVE=false

GETDATA=false


while [ "$1" != "" ]; do
    case $1 in
        20NG | 20ng | todosC )
            echo "Argument includes corpus 20NG"
            TWENTYNG=true
            ;;
        r | reuters | todosC )
            echo "Argument includes corpus reuters"
            REUTERS=true
            ;;
        w | wiki | wikipedia | todosC )
            echo "Argument includes corpus wikipedia"
            WIKIPEDIA=true
            ;;



        smh | todosM )
            echo "Argument includes smh embbedings"
            SMH=true
            ;;
        CTXT | ctxt | context | todosM )
            echo "Argument includes context embbedings"
            CTXT=true
            ;;
        w2v | word2vec | word | todosM )
            echo "Argument includes word2vec embbedings"
            W2V=true
            ;;
        glove | todosM )
            echo "Argument includes glove embbedings"
            GLOVE=true
            ;;


        data | download )
            echo "Will download corpus."
            GETDATA=true
            ;;

        \? )
            echo "Specify which corpus to use (20NG, reuters, wiki, todosC), \n \
            and which vector encoding to use (smh, context, word2vec, glove, todosM)"
            ;;
    esac        
    shift
done    



# echo "Checking variables status."
# echo
# echo
# if $SMH && $TWENTYNG ; then
#     echo "si en el if"
# fi
# echo $TWENTYNG && $SMH
# echo [ $SMH ] && [ $TWENTYNG ]
# echo $SMH
# echo $TWENTYNG


#####################################
#   Stop Words

if [ ! -f $DATAPATH/stopwords_english.txt ] ; then
    echo "Downloading stopwords"
    wget -qO- -O $DATAPATH/stopwords_english.txt \
         https://raw.githubusercontent.com/pan-webis-de/authorid/master/data/stopwords_english.txt
fi



####################################
#   Preloaded Encoders


if $GLOVE && [ ! -f $THIRDPARTYPATH/glove.6B.zip ] ; then
    if [ ! -f $THIRDPARTYPATH ] ; then
        mkdir $THIRDPARTYPATH
        echo "En el if mal"
    fi

    echo "Downloading pre-trained Glove word embeddings from Stanford s website."
    wget  -O $GLOVENAME "http://nlp.stanford.edu/data/glove.6B.zip"
    mkdir $THIRDPARTYPATH/glove.6B.2
    unzip -d $THIRDPARTYPATH/glove.6B.2/ $GLOVENAME
    # unzip -d `pwd`/glove.6B.2/ glove.6B.50d.zip
    echo "Glove unziped."
fi








###################################
#   Download and process specified corpuses


if $TWENTYNG ; then
    mkdir -p $DATAPATH/20newsgroups
fi


#   20NG and SMH

if $SMH && $TWENTYNG ; then
    echo
    echo
    echo "SMH 20NG "
    echo
    echo
    echo "Generating 20 newsgroups SMH reference text (tokenized)"
    python python/corpus/20ng2ref.py $DATAPATH/20newsgroups

    echo "Generating BOWs from 20 newsgroups reference text"
    python python/corpus/ref2corpus.py $DATAPATH/20newsgroups/20newsgroups.ref \
        $DATAPATH/stopwords_english.txt \
        $DATAPATH/20newsgroups/ \
        -c 40000
    
    echo "Genereting inverted file from corpus"
    smhcmd ifindex $DATAPATH/20newsgroups/20newsgroups40000.corpus $DATAPATH/20newsgroups/20newsgroups40000.ifs
    
    echo "Done processing 20 newsgroups corpus"
    echo
    echo
fi


if $TWENTYNG && $GLOVE ; then
    echo
    echo
    echo "20NG with glove embbedings"
    echo
    echo
    python python/corpus/20ng2glove.py glove

    echo "Finished training model."

fi









if $REUTERS; then
    echo "Preparing Reuters"
    mkdir -p $DATAPATH/reuters
    echo -n "Enter path of Reuters dataset: "
    read REUTERSPATH

    echo "Genereting reference text from Reuters directory $REUTERSPATH"
    python $ROOTPATH/python/corpus/reuters2ref.py \
	     $REUTERSPATH \
	     $DATAPATH/reuters/

    for VOCSIZE in 20000 40000 60000 80000 100000
    do
        echo "Genereting BOWs (vocabulary size = $VOCSIZE) from Reuters reference"
        python $ROOTPATH/python/corpus/ref2corpus.py \
	         $DATAPATH/reuters/reuters.ref \
	         $DATAPATH/stopwords_english.txt \
	         $DATAPATH/reuters/ \
	         -c $VOCSIZE

        echo "Genereting inverted file from corpus"
        smhcmd ifindex $DATAPATH/reuters/reuters$VOCSIZE.corpus $DATAPATH/reuters/reuters$VOCSIZE.ifs
    done
    
    echo "Done processing Reuters corpus"
fi

if $WIKI2TEXT; then
    if ! command -v nim >/dev/null 2>&1; then 
       mkdir -p $THIRDPARTYPATH

       echo "Installing Nim"
       git clone git://github.com/nim-lang/Nim.git $THIRDPARTYPATH/Nim
       cd $THIRDPARTYPATH/Nim
       git clone --depth 1 git://github.com/nim-lang/csources
       cd csources
       sh build.sh
       cd ..
       bin/nim c koch
       ./koch boot -d:release
       export PATH=$PATH:$THIRDPARTYPATH/Nim/bin

       echo "Done installing Nim"
   fi

    echo "Installing wiki2text"
    git clone https://github.com/rspeer/wiki2text.git $THIRDPARTYPATH/wiki2text
    cd $THIRDPARTYPATH/wiki2text
    make
fi

if $WIKIPEDIA; then
    echo "Preparing Wikipedia"    
    mkdir -p $DATAPATH/wikipedia

    if [ ! -f $DATAPATH/wikipedia/wikien.xml.bz2 ]; then
        echo "Downloading Wikipedia dump"
        wget -qO- -O $DATAPATH/wikipedia/wikien.xml.bz2 \
             $WIKIDUMPEN
    fi
    
    echo "Uncompressing and parsing Wikipedia dump"
    bunzip2 -c $DATAPATH/wikipedia/wikien.xml.bz2 \
        | $THIRDPARTYPATH/wiki2text/wiki2text > $DATAPATH/wikipedia/enwiki.txt

    echo "Genereting reference text from wikipedia"
    python $ROOTPATH/python/corpus/wiki2ref.py \
	   $DATAPATH/wikipedia/enwiki.txt \
	   $DATAPATH/wikipedia/
    
    echo "Genereting BOWs from wikipedia reference"
    python $ROOTPATH/python/corpus/ref2corpus.py \
	   $DATAPATH/wikipedia/enwiki.ref \
	   $DATAPATH/stopwords_english.txt \
	   $DATAPATH/wikipedia/ \
	   -c 1000000

    echo "Generating inverted file from corpus"
    smhcmd ifindex $DATAPATH/wikipedia/enwiki1000000.corpus $DATAPATH/wikipedia/enwiki.ifs

    echo "Using Wikipedia as reference for computing NPMI scores"
    mkdir -p $DATAPATH/ref
    cp $DATAPATH/wikipedia/enwiki.ref $DATAPATH/ref/
    
    echo "Done processing Wikipedia corpus"
fi
