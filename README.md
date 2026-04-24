### Usage

1. Use the command "make" to run the Makefile and install dependencies
2. Go to preprocessing.ipynb and run each of the cells in order
3. To see the model in action, run the cells in model.ipynb in order (you may want to skip the cell with the grid search)
4. To TEST, run "pytest -m pytest test_model.py" in the terminal

## Preprocessing
Before building the model, I first built my dataset pulling from both kaggle and nltk word datasets. I then shifted these all into one csv file which had the words in one column and their part of speeches in a second column. After this, I removed all of the words with any strange symbols such as numbers, dashes, slashes, or punctuation marks. Finally, I used the BERT encoder model from HuggingFace to convert all of these words into their respective embeddings, added meaningless embeddings to add onto the end of the shorter words (this way the model would take all input tensors of the same shape), and created attention masks so that the model would not accidentally attend to these meaningless padding vectors.

## The Initial Model
This model I built is a single-layer transformer using an MLP with one hidden layer with n layers and a  multi-headed self-attention (MHA) with m heads without causal masking allowing embeddings to attend both forwards and backwards. The experimentation I plan on doing includes finding the optimal number of dimensions in hidden layers, number of attention heads, the number of hidden layers, and using other techniques such as dropout to decrease overfitting. I am using the Adam optimizer for the optimizer function using L2 regularization during training and using cross entropy loss as a loss function as well.
