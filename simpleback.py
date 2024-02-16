# %%


from simplehelp import *
import torch
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import argparse
import json
import os
from collections import Counter
import random


# os.environ["WANDB_API_KEY"] = "4c2a9ff74fdb68f1f92a87d2ff834315f06a3530"
# os.environ["WANDB_SILENT"] = "true"
# wandb.login()


# %%
# specify device
cuda_available = torch.cuda.is_available()
if cuda_available:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %%
from transformers import GPT2Tokenizer, GPT2Model

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
# embeddings = model.transformer.wte.weight.detach()
embeddings = model.get_input_embeddings().weight.detach()
model.to(device)
embeddings.to(device)


# %%
target_output = " world"
output_ix = tokenizer.encode(target_output, return_tensors="pt")[0].to(device)
# %%


def optimise_input(
    model,
    word_embeddings,
    tokenizer,
    device,
    epochs=100,
    lr=0.1,
    batch_size=20,
    input_len=10,
    target_output=" world",
    output_len=None,
    dist_reg=0.1,
    perp_reg=0,
    loss_type="log_prob_loss",
    optimiser="Adam",
    **kwargs
):
    output_ix = tokenizer.encode(target_output, return_tensors="pt")[0].to(device)
    word_embeddings = word_embeddings / torch.sqrt(
        torch.sum(word_embeddings**2, dim=-1, keepdim=True)
    )

    if output_len is None or output_len < output_ix.shape[0]:
        output_len = output_ix.shape[0]

    num_clusters = batch_size * input_len
    _, centroids = kkmeans(word_embeddings.detach(), num_clusters)

    #
    start_input = centroids.reshape(batch_size, input_len, -1)

    input = torch.nn.Parameter(start_input.to(device), requires_grad=True)
    optimiser = torch.optim.Adam([input], lr=lr, eps=0.0001)

    for e in range(epochs):
        norm_input = input / torch.sqrt(torch.sum(input**2, dim=-1, keepdim=True))
        logits, emb, perp = model_emb(model, norm_input, word_embeddings, output_len)
        probs = torch.softmax(logits, dim=-1)
        perp_loss = perp.mean()

        if output_len > output_ix.shape[0]:
            target_logits = logits[:, :, output_ix].max(dim=1)[0]
            target_probs = probs[:, :, output_ix].max(dim=1)[0]
        else:
            target_logits = logits[:, torch.arange(output_len), output_ix]
            target_probs = probs[:, torch.arange(output_len), output_ix]

        token_dist, closest_ix = [], []
        for b in norm_input:
            tds, cixs = [], []
            for be in b:
                _, cix, td, _ = closest_tokens(be, word_embeddings, tokenizer)
                tds.append(td)
                cixs.append(cix)
            token_dist.append(torch.stack(tds))
            closest_ix.append(torch.stack(cixs))

        mean_token_dist = torch.stack(token_dist).squeeze(-1).mean()

        loss = -torch.log(target_probs)

        batch_loss = loss.mean()
        total_loss = torch.stack(
            [mean_token_dist * dist_reg, batch_loss, perp_loss * perp_reg]
        ).mean()

        model_outs = model.generate(closest_ix, max_length=output_len + input_len)
        optimised_inputs = set()
        for b in range(batch_size):
            if target_output in tokenizer.decode(model_outs[b][input_len:]):
                optimised_inputs.add(tokenizer.decode(model_outs[b]))

        optimiser.zero_grad()
        total_loss.backward(retain_graph=True)
        optimiser.step()

    return {"Optimised Inputs": list(optimised_inputs)}


# %%
optimise_input(model, embeddings, tokenizer, device)
# specify modle and run

# %%
