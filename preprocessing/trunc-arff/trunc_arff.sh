#!/bin/bash

ARFF_LENGTH=3
INPUT=ecbdl14-train.csv.gz
OUTPUT=ecbdl14-train-edited.csv.gz

echo "Starting..."
START_TIME=$(date +%s)

gunzip -c $INPUT | tail -n+$ARFF_LENGTH | gzip > $OUTPUT

END_TIME=$(date +%s)

echo "Completed in" $((END_TIME - START_TIME)) "seconds"
