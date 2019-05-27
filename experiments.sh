#!/bin/bash
BEAM_SIZE=10
MAX_SEQ_LENGTH=50

# Beam Search
python examples/run_gpt2.py \
--inputs_file samples/webtext-test-prompts.txt \
--nsamples="$BEAM_SIZE" \
--length="$MAX_SEQ_LENGTH" \
--do_beam_search=True

# Random sampling
python examples/run_gpt2.py \
--inputs_file samples/webtext-test-prompts.txt \
--nsamples="$BEAM_SIZE" \
--length="$MAX_SEQ_LENGTH" \
