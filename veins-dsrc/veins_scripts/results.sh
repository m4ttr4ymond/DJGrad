#!/bin/bash

SIMULATION=$1
INPUT_FILE=$2

RESULTS_FOLDER="../simulations/${SIMULATION}/results"
EXTENSION=${INPUT_FILE: -3}
OUTPUT_FILE=${RESULTS_FOLDER}/${INPUT_FILE/".${EXTENSION}"/"-${EXTENSION}"}.csv

case ${INPUT_FILE: -5:-4} in
  i)
    INTERACTION="2s Interaction"
    ;;
  s)
    INTERACTION="1s Interaction"
    ;;
  *)
    echo "INPUT ERROR"
    exit 1
esac

if [[ $INPUT_FILE == *"Drop"* ]]; then
  DROP="90% Packet Drop"
else
  DROP="No Packet Drops"
fi

# Change scenario to Title Case
TITLE=$(echo ${SIMULATION%%"_small"} | sed -r 's/(^|_)([a-z])/ \U\2/g')
TITLE="${TITLE} Scenario - ${INTERACTION} - ${DROP}"

case ".${EXTENSION}" in
  ".sca")
    ./sca2csv.pl -F gradientCount ${RESULTS_FOLDER}/${INPUT_FILE} > ${OUTPUT_FILE}
    python3 sca-results.py ${OUTPUT_FILE}
    ;;
  ".vec")
    ./vec2csv.pl -F gradientCount ${RESULTS_FOLDER}/${INPUT_FILE} > ${OUTPUT_FILE}
    python3 vec-results.py ${OUTPUT_FILE} "${TITLE}"
    ;;
  *)
    echo "Must give .sca or .vec file"
    exit 1
    ;;
esac
