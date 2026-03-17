#!/bin/bash

if [ $# -lt 3 ]
then
        echo "Usage: $0 <index dir> <qrels file> <query file>"
        exit
fi

INDEX_DIR=$1
QRELS_FILE=$2
QUERY_FILE=$3

mvn exec:java -Dexec.mainClass="Retriever" -Dexec.args="$INDEX_DIR $QRELS_FILE $QUERY_FILE"
