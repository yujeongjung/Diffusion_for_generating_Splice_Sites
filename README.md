# Generating-Splice-Site-using-Diffusion-Model
> IRTP project in RC4 related to creating Arabidopsis thaliana DNA sequences with correct splice site using Diffusion Model
Used model: Diffusion Denoising Probabilistic Model (DDPM)

Generate synthetic DNA sequences using Diffusion Model trained with real dataset.

## Dataset overview: Species: Arabidopsis thaliana

SPC acceptor positive (9310)
SPC acceptor negative (277255)
SPC donor positive (9208)
SPC donor negative (263507)
Each sequence length: 402

## Evaluation method

Train the model using real dataset and test the model with real dataset
Train the model using synthetic dataset and test the model with real dataset
Train the model using real dataset and test the model with synthetic dataset
Train the model using synthetic dataset and test the model with synthetic dataset
