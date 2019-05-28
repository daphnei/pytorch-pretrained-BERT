#!/usr/bin/env python3
import codecs
import argparse
import logging
from tqdm import trange
import json
import torch
import torch.nn.functional as F
import numpy as np
import beam_search

from torch.nn.functional import log_softmax
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def do_beam_search(model, length, start_token=None, batch_size=None, context=None, beam_size=10, device='cuda'):
    def decode_step(seq_so_far, current_state, k, feed_all_timesteps=False, remove_unknowns=False,
                    get_attention=False, device_ids=None):
        if get_attention or remove_unknowns or feed_all_timesteps:
            raise ValueError('This hacky code doesn\'t support any of these options.')

        if not isinstance(seq_so_far, list):
            seq_so_far = [seq_so_far]
        if current_state == None:
            current_state = len(seq_so_far) * [None]
        assert(len(seq_so_far) == len(current_state))

        # Iterate over all of the possible beams and predict a next token for each.
        # Ideally, this code would be batched in some way, but I am too lazy to
        # implement that.
        all_new_states = []
        all_logprobs = []
        all_words = []
        for seq, state in zip(seq_so_far, current_state):
            if state is not None:
              # Model only expects the sequence tokens it has not seen before.
              seq = seq[:, -1].unsqueeze(0)
            logits, state = model(seq, past=state)
            logits = logits.select(1, -1).contiguous()
            logprobs = log_softmax(logits, dim=1)
            logprobs, words = logprobs.topk(k, 1) 
            all_new_states.append(state)
            all_logprobs.append(logprobs)
            all_words.append(words)

        all_logprobs = torch.cat(all_logprobs, 0)
        all_words = torch.cat(all_words, 0)
        # all_words is [batch_size x current_seq_length] containing the IDs in each beam.
        # all_logprobs is [batch_size x current_seq_length] score for each word in each beam.
        # all_new_states is length barch_size list of the model state for each beam.
        return all_words, all_logprobs, all_new_states
    
    seq_gen = beam_search.SequenceGenerator(
            decode_step=decode_step,
            beam_size=beam_size,
            max_sequence_length=length,
            get_attention=False,
            length_normalization_factor=0,
            device_ids=device)

    context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    results = seq_gen.beam_search(context, None) 
    
    for result in results:  # Iterates over items in the batch.
        for beam_output in result:  # Iterates over beams in the example.
            yield beam_output.output, beam_output.score.tolist()

def sample_sequence(model, length, start_token=None, batch_size=None,
                    context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def decode_interactive(model, enc, device, args):
    while True:
        context_tokens = []
        if not args.unconditional:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=context_tokens,
                    start_token=None,
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:, len(context_tokens):].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)
        else:
            generated = 0
            for _ in range(args.nsamples // args.batch_size):
                out = sample_sequence(
                    model=model, length=args.length,
                    context=None,
                    start_token=enc.encoder['<|endoftext|>'],
                    batch_size=args.batch_size,
                    temperature=args.temperature, top_k=args.top_k, device=device
                )
                out = out[:,1:].tolist()
                for i in range(args.batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                    print(text)
            print("=" * 80)


def decode_from_file(model, enc, device, args):
    results = []
    with codecs.open(args.inputs_file, 'r', encoding='utf-8') as f:
        for example_idx, line in enumerate(f):
            print('Starting decoding for example %d' % (example_idx))
            raw_text = line.strip()
            context_tokens = enc.encode(raw_text)
            generated = 0
            if args.do_beam_search:
                outputs = do_beam_search(
                    model=model, length=args.length,
                    context=context_tokens,
                    start_token=None, batch_size=args.batch_size,
                    beam_size=10, device=device)
            else:
                outputs = []
                for _ in range(args.nsamples // args.batch_size):
                    out = sample_sequence(
                        model=model, length=args.length,
                        context=context_tokens,
                        start_token=None,
                        batch_size=args.batch_size,
                        temperature=args.temperature, top_k=args.top_k, device=device
                    )
                    for o in out:
                      outputs.append((o, -1))
                      generated += 1

            # Save results into our JSON format.
            result = {"input": raw_text, "pred": [], "scores": []}
            for out, score in outputs:
                out = out[len(context_tokens):].tolist()
                text = enc.decode(out)
                result["pred"].append(text)
                result["scores"].append(score)
            results.append(result)
    with codecs.open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump({"result": results}, f)


def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--inputs_file', type=str, default=None)
    parser.add_argument('--output_file', type=str, default='results.json')
    parser.add_argument('--do_beam_search', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    if args.batch_size == -1:
        args.batch_size = 1
    assert args.nsamples % args.batch_size == 0

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.eval()

    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)

    if args.inputs_file is None:
      decode_interactive(model, enc, device, args)
    else:
      decode_from_file(model, enc, device, args)

if __name__ == '__main__':
    run_model()


