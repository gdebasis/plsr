#!/bin/bash

if [ $# -lt 4 ]
then
        echo "Usage: $0 <collection file> <index folder> <query file> <query index folder>"
        exit
fi

mvn exec:java -Dexec.mainClass="MsMarcoIndexer" -Dexec.args="$1 $2 $3 $4"
