#!/bin/bash

INPUT=ecbdl14-train.csv.gz
OUTPUT=ecbdl14-train-sample.csv.gz
SAMPLE_SIZE=3500000

echo "Starting..."
START_TIME=$(date +%s)

gunzip -c $INPUT | shuf -n $SAMPLE_SIZE | gzip > $OUTPUT

END_TIME=$(date +%s)

echo "Completed in" $((END_TIME - START_TIME)) "seconds"

