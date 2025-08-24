# Generating-Splice-Site-using-Diffusion-Model
> IRTP project in RC4 related to creating Arabidopsis thaliana DNA sequences with correct splice site using Diffusion Model
Used model: Diffusion Denoising Probabilistic Model (DDPM)

Generate synthetic DNA sequences using Diffusion Model trained with real dataset.

## Dataset overview: Species: Arabidopsis thaliana

SPC acceptor positive (9310) \n
SPC acceptor negative (277255) \n
SPC donor positive (9208) \n
SPC donor negative (263507) \n
Each sequence length: 402

## Evaluation method

Train the model using real dataset and test the model with real dataset \n
Train the model using synthetic dataset and test the model with real dataset\n
Train the model using real dataset and test the model with synthetic dataset\n
Train the model using synthetic dataset and test the model with synthetic dataset
