from .helpers import *
import torch
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import argparse
import json
import os
from collections import Counter
import random


def opt_input(model):
    output_ix = tokenizer.encode(target_output, return_tensors="pt")[0].to(device)
