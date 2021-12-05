#!/bin/bash

SIMULATION=$1
RESULTS_FOLDER="../simulations/${SIMULATION}/results"
OUTPUT_FILE=${RESULTS_FOLDER}/Default-#0-sca.csv

./sca2csv.pl -F gradientCount ${RESULTS_FOLDER}/Default-#0.sca > ${OUTPUT_FILE}

python3 sca-results.py ${OUTPUT_FILE}
