#!/bin/bash

SIMULATION=$1
RESULTS_FOLDER="../simulations/${SIMULATION}/results"
OUTPUT_FILE=${RESULTS_FOLDER}/Default-#0-vec.csv

./vec2csv.pl -F gradientCount ${RESULTS_FOLDER}/Default-#0.vec > ${OUTPUT_FILE}

python3 vec-results.py ${OUTPUT_FILE}
