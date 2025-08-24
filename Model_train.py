from Model_test import SimpleUNet, DDPM
from sklearn.model_selection import train_test_split
import torch
import os
import pandas as pd
import numpy as np
from one_hot import OneHotEncodeTransform
from torch.utils.data import DataLoader
from dna_dataset import DNADataset
from sklearn.metrics import f1_score


# Hyperparameters
epochs = 10
betas = np.linspace(1e-4, 0.02, 1000)
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 32
print(f"GPU operation: {device}")

# Model, optimizer, and loss function
input_dim = 4  # DNA sequence one-hot encoded dimension
unet = SimpleUNet(input_dim, 64).to(device)
ddpm = DDPM(betas, unet).to(device)
optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)


path = os.path.join('REAL_dataset', 'clean_acceptors.pos')

with open(path, 'r') as file:
    lines = file.readlines()
    
df1 = pd.DataFrame(lines, columns=['DNA_sequence'])
df = df1.replace('\n', '', regex=True)  # Remove newline characters from the entire DataFrame

# Extract sequences and labels
sequences = df['DNA_sequence'].tolist()

# Split the dataset
train_seqs, temp_seqs = train_test_split(sequences, test_size=0.2, random_state=42)
valid_seqs, test_seqs = train_test_split(temp_seqs, test_size=0.5, random_state=42)

# Define the transform
onehot_transform = OneHotEncodeTransform()

# Create the datasets
train_dataset = DNADataset(train_seqs, transform=onehot_transform)
valid_dataset = DNADataset(valid_seqs, transform=onehot_transform)
test_dataset = DNADataset(test_seqs, transform=onehot_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(epochs):
    for step, x in enumerate(train_loader):
        x = torch.tensor(x, dtype=torch.float32).to(device)
        t = torch.randint(0, len(betas), (x.size(0),), device=device).long()

        loss = ddpm.p_losses(x, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")


def max_element(tensor):

    # Get the indices of the maximum values in each column
    _, max_indices = torch.max(tensor, dim=0)

    # Create a binary tensor with the same shape as the original tensor
    binary_tensor = torch.zeros_like(tensor)

    # Use the indices to set the maximum values to 1
    cols = torch.arange(tensor.size(1))  # Column indices
    binary_tensor[max_indices, cols] = 1
    return binary_tensor

def transform_and_combine(matrix):
    # Define the nucleotide mapping
    nucleotide_map = {
        0: '',    # For the positions where there is no '1'
        1: 'A',   # First row maps to 'A'
        2: 'T',   # Second row maps to 'T'
        3: 'G',   # Third row maps to 'G'
        4: 'C'    # Fourth row maps to 'C'
    }
    
    # Initialize an empty list to store the combined sequence
    combined_sequence = []
    
    # Iterate over each column in the matrix
    for col in matrix.T:  # Transpose to iterate over columns
        # Find the index of '1' in the row
        for i, value in enumerate(col):
            if value == 1:
                # Append the corresponding nucleotide to the sequence
                combined_sequence.append(nucleotide_map[i + 1])
                break
    
    # Join the list to form the final sequence
    return ''.join(combined_sequence)

seq_length = 402

# Sampling
@torch.no_grad()  # no need of using gradient
def generate_dna_sequences(ddpm, num_sequences, seq_length, device):
    new_df = pd.DataFrame({'DNA_sequence': []})
    samples = ddpm.sample(seq_length=seq_length, batch_size=num_sequences, device=device)
    for idx, sample in enumerate(samples):
        sample = max_element(sample)
        dna_decode = transform_and_combine(sample)
        new_df.loc[idx+1] = dna_decode
    return new_df

num_novel_sequences = 5
novel_df = generate_dna_sequences(ddpm, num_novel_sequences, seq_length, device)


def compare_sequences(seq1, seq2):
    return [1 if s1 == s2 else 0 for s1, s2 in zip(seq1, seq2)]

# F1 score function
def calculate_f1_score(true_sequences, pred_sequences):
    y_true = []
    y_pred = []
    
    for true_seq, pred_seq in zip(true_sequences, pred_sequences):
        y_true.extend(compare_sequences(true_seq, true_seq))
        y_pred.extend(compare_sequences(true_seq, pred_seq))
    
    return f1_score(y_true, y_pred)

ori = df.head(5)
true_sequences = df['DNA_sequence']
pred_sequences = novel_df['DNA_sequence']

# F1 score calculation
f1 = calculate_f1_score(true_sequences, pred_sequences)
print(f"F1 Score: {f1:.4f}")

