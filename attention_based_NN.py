import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

np.random.seed(42)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from tqdm.auto import tqdm, trange
from collections import Counter
import random
from torch import optim

import pandas as pd
import pickle

import wandb

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import seaborn as sns

# Sort of smart tokenization
from nltk.tokenize import RegexpTokenizer
from gensim.models import KeyedVectors

# Attention plotting
import matplotlib.pyplot as plt

### PROBLEM 17

class AttentionNNClassifier(nn.Module):

    def __init__(self, word2vec_model:KeyedVectors, num_heads, device, embedding_size=100):
        '''
        Creates the new classifier model. embeddings_fname is a string containing the
        filename with the saved pytorch parameters (the state dict) for the Embedding
        object that should be used to initialize this class's word Embedding parameters
        '''
        super(AttentionNNClassifier, self).__init__()

        # Save the input arguments to the state
        self.word2vec_model = word2vec_model
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        # attention layers
        self.attention_heads = nn.Parameter(torch.randn(self.num_heads, self.embedding_size)) # [num_heads, embedding_size]
        # output layer
        self.output_layer = nn.Linear(self.embedding_size * self.num_heads, 1)
        self.device = device

    def forward(self, texts):
        # Get the word embeddings for the ids
        word_embeddings = torch.tensor(np.array([[self.word2vec_model[int(word)] for word in text] for text in texts])).to(self.device) # [batch_size, seq_length, embedding_size]

        # Calcuate the 'r' vectors which are the dot product of each attention head
        # with each word embedding. You should be getting a tensor that has this
        # dot product back out---remember this vector is capturing how much the
        # head thinks the vector is relevant for the task
        r_vectors = self.attention_heads @ word_embeddings.transpose(1,2) # [num_heads, embedding_size] x [batch_size, embedding_size, seq_length] = [batch_size, num_heads, seq_length]


        # Calcuate the softmax of the 'r' vector, which call 'a'. This will give us
        # a probability distribution over the tokens for each head. Be sure to check
        # that the softmax is being calculated over the right axis/dimension of the
        # data (You should see probability values that sum to 1 for each head's
        # ratings across all the tokens)
        a_vectors = F.softmax(r_vectors, dim=2)  # [batch_size, num_heads, seq_length]


        # Calculate the re-weighting of the word embeddings for each head's attention
        # weight and sum the reweighted sequence for each head into a single vector.
        # This should give you n_heads vectors that each have embedding_size length.
        # Note again that each head should give you a different weighting of the
        # input word embeddings
        weighted_embeddings = torch.bmm(a_vectors, word_embeddings) # [batch_size, num_heads, embedding_size]


        # Create a single vector that has all n_heads' attention-weighted vectors
        # as one single vector. We need this one-long-vector shape so that we
        # can pass all these vectors as input into a layer.
        #
        # NOTE: if you're doing Option 2 for representing attention, you don't
        # actually need to create a new vector (which is very inefficient).
        # Instead, you can create a new *view* of the same data that reshapes the
        # different heads' vectors so it looks like one long vector.
        concatenated_vectors = weighted_embeddings.reshape(-1) # [batch_size(1)*num_heads*embedding_size]

        # Pass the side-by-side attention-weighted vectors through your linear
        # layer to get some output activation.
        #
        # NOTE: if you're feeling adventurous, try adding an extra layer here
        # which will allow you different attention-weighted vectors to interact
        # in making the model decision
        output_activations = self.output_layer(concatenated_vectors)

        # Return the sigmoid of the output activation *and* the attention
        # weights for each head. We'll need these later for visualization
        predictions = torch.sigmoid(output_activations)

        return predictions, a_vectors