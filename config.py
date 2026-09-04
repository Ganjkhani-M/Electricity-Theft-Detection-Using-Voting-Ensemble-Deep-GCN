# This is the configuration file for the Hybrid deep GCN + Voting Ensemble model
# Important Notes:
# Please change the 'csv_path' based on your local machine or get it from google colab when you upload it on google colab
# fraction represents the fraction of data that can be used during the training process. For instance, fraction = 1.0 means that we are considering all of the available data during our training process
# training_ratio represents the amount of data that is used during the training process. For instance, training_ratio = 0.7 , test_ratio = 0.3

import os

##### Data Settings 
csv_path = '/content/Electricity.csv' 
fraction = 1.0 
training_ratio = 0.7


#### GCN Model Settings 
k_neighbors = 70 # KNN with k=70
hidden_dim = 32 # Dimension of the hidden fully connected layer
embedding_dim = 32 # Dimension of the embedding (The GCN's output)
GCN_layers = 12 # Number of GCN layers
learning_rate = 1e-4 # Learning rate
weight_decay = 1e-4  
epochs = 150
patience = 30


#### Voting Ensemble Settings
oversample_factor = 7
MAP_K = 100


#### Output Settings
save_results = True
results_file = 'results/learning_curve_results_k_study.csv'
print_loss_every = 5
plot_loss = True

