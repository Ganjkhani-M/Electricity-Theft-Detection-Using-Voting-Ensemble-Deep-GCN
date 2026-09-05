# Weighted Voting Ensemble Classifier. 

import numpy as np 
from sklearn.ensemble import RandomForestClassifier , VotingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier 

from .utils import oversample_features , map_at_k

def train_voting_ensemble(x_train , y_train , x_val , y_val , factor = 7, map_k = 100):
  x_bal , y_bal = oversample_features(x_train , y_train , factor)

  gb1 = HistGradientBoostingClassifier(max_depth = 6, learning_rate = 0.03, max_iter = 300 , min_samples_leaf = 50 , random_state = 42)
  gb2 = HistGradientBoostingClassifier(max_depth = 8, learning_rate = 0.05, max_iter = 300 , min_samples_leaf = 50 , random_state = 123)
  gb3 = HistGradientBoostingClassifier(max_depth = 5, learning_rate = 0.02, max_iter = 500 , min_samples_leaf = 50 , random_state = 999)
  rf = RandomForestClassifier(n_estimators = 500 , max_depth = 15, min_samples_leaf = 10, class_weight = 'balanced', random_state = 42, n_jobs = -1)

  models = [gb1, gb2, gb3, rf]

  # Train each of the models in voting ensemble
  for m in models:
    m.fit(x_bal, y_bal)

  # Calculating MAP@K on Validation
  val_probs = np.column_stack([m.predict_proba(x_val)[:, 1] for m in models])
  maps = [map_at_k(y_val, val_probs[:, i], map_k) for i in range(4)]
  weights = np.array(maps) / (np.sum(maps) + 1e-8)

  # Bulding the Voting Classifier using calculated weights
  voting = VotingClassifier(
              estimators = [('gb1', gb1) , ('gb2', gb2) , ('gb3', gb3) , ('rf', rf)], 
              voting = 'soft',
              weights = weights)
  voting.fit(x_bal, y_bal)

  return voting 
  
  
