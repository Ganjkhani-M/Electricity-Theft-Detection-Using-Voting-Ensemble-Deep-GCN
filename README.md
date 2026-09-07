# Electricity Theft Detection using Hybrid Deep GCN and Weighted Voting Ensemble 

This Project presents a hybrid model for identifying fraudulent users ( electricity theft) using daily consumption data. The proposed architecture consists of three main stages:
1. User-User Graph Construction: Using the KNN algorithm and cosine distance to discover similar consumption patterns.

2. Deep Embedding Extraction: Using a 12-layer GCN (No Residual Connections) that models structural relationships between users.

3. Final Classification: Using a Weighted Voting Ensemble consisting of three HistGradientBoosting model and one RandomForest, where the weight of each model is determined based on MAP@100 on the validation data.



## Key Features are as follows:
1: Robust Preprocessing : Removing samples with over 50% zeros, linear interpolation, 2-sigma clipping, and MinMax normalization. 

2: Feature Engineering: Extracting 19 statical, frequency-domain (FFT), and trend features from the time series.

3: Handling Imbalance: Oversampling with a factor of 7 for the positive class + positive weighting in the GCN loss function. 

4: Smart Ensemble Weighting: Using the MAP@K metric instead of Accuracy to prioritize the detection of fraudulent users at higher ranks. 



## Code Structure 
1. config.py : Main configuration (data path, graph , hyper parameters).
2. src/preprocess.py : Data Cleaning and feature extraction.
3. src/graph_builder.py : Similarity graph construction.
4. src/models.py : 12-layer GCN architecture.
5. src/trainer.py : Training loop with Early stopping.
6. src/ensemble.py : Voting Ensemble training and weighting.
7. src/utils.py : Utility functions.
8. main.py : Entry point to run the entire code.


## How we can run the entire code?
1. Install the required libraries:
   pip install -r requirements.txt
2. Set the dataset CSV path in the 'config.py'.
3. Execute the main script:
   python main.py


## Results
Results will saved in 'results/learning_curve_results_k_study.csv', and the loss curve plot is stored in the 'results/' folder. 
The primary evaluation metrics are AUC, MAP@100, and MAP@200. 
   


