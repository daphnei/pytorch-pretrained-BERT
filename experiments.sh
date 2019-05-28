#!/bin/bash
BEAM_SIZE=10
MAX_SEQ_LENGTH=50

OUTPUT_DIR="experiments/experiments_$BEAM_SIZE"

mkdir $OUTPUT_DIR

# Beam Search
python -m pdb examples/run_gpt2.py \
--inputs_file samples/webtext-test-prompts.txt \
--nsamples="$BEAM_SIZE" \
--length="$MAX_SEQ_LENGTH" \
--do_beam_search=True \
--output_file="${OUTPUT_DIR}/standard_beam.json"

# Random sampling
python examples/run_gpt2.py \
--inputs_file samples/webtext-test-prompts.txt \
--nsamples="$BEAM_SIZE" \
--length="$MAX_SEQ_LENGTH" \
--output_file="${OUTPUT_DIR}/random_sampling.json"
