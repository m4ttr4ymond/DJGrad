#!/bin/bash

SIMULATION=$1
INPUT_FILE=$2

RESULTS_FOLDER="../simulations/${SIMULATION}/results"
EXTENSION=${INPUT_FILE: -3}
OUTPUT_FILE=${RESULTS_FOLDER}/${INPUT_FILE/".${EXTENSION}"/"-${EXTENSION}"}.csv

# Change scenario to Title Case
TITLE=$(echo ${SIMULATION} | sed -r 's/(^|_)([a-z])/ \U\2/g')

case ".${EXTENSION}" in
  ".sca")
    ./sca2csv.pl -F gradientCount ${RESULTS_FOLDER}/${INPUT_FILE} > ${OUTPUT_FILE}
    python3 sca-results.py ${OUTPUT_FILE}
    ;;
  ".vec")
    ./vec2csv.pl -F gradientCount ${RESULTS_FOLDER}/${INPUT_FILE} > ${OUTPUT_FILE}
    python3 vec-results.py ${OUTPUT_FILE} "${TITLE} Scenario"
    ;;
  *)
    echo "Must give .sca or .vec file"
    exit 1
    ;;
esac
