# CS506-Final-Project

For this project, I plan on creating a classification model to identify what type part of speech a word is based on its embedding.

# Timeline:
- February 11 - Submit Proposal
- Febraury 20 - Finish deciding on which dataset to actually implement this project with
- February 25 - Complete Data cleaning and restructuring (the dataset I - found is structured in a strange way)
 -March 1 - Start testing various models various models and evaluating what I plan on using for my final results (most likeley simple MLP)
- March 15 - Finish testing and researching ML methods and focus on finding the best set of hyperparamters for the model for my final prject (this may mean testing several times with several different configurations)
- March 25 - Finish making visualizations for projects
- April 1 - Finish writing reports and creating presentation

# Project Goals:
- My model must be able to correctly identify what part of speech a word is with greater than 80% accuracy
- Myu presentation, visualization, and reports must be able to be clearly and easily understood

# Data Collection Plan:
- I plan on initially using a dataset I found on kaggle which maps English words to their respective parts of speech (https://www.kaggle.com/datasets/thedevastator/common-english-parts-of-speech?select=adjectives.csv)
- I will then need to use some method of changing the words to their embeddings either by Gensim or by using the encoder functions from the Transformers Library via HuggingFace (perhaps different models will encode this information in different ways and it may make sense to try a few)
