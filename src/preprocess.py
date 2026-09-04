# This file is related to the data preprocessing section including cleaning data, and feature engineering

# import packages 
import numpy as np
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler


def preprocess_sgcc_data(csv_path):
  # importing the data, and removing data which more than 50% of them are zero
  df = pd.read_csv(csv_path)
  y = df.iloc[:,1].values.astype(np.float32)
  x_raw = df.iloc[:, 2:].values.astype(np.float32)

  # adding one zero to the end of the x_raw its length was 1034, to be 1035
  if x_raw.shape[1] == 1034:
    x_raw = np.pad(x_raw, ((0,0), (0,1)), constant_values=0)


  # Linear interpolation
  def linear_interpolate(series):
    series = series.copy()
    nan_mask = np.isnan(series)
    if not nan_mask.any():
      return series

    idx = np.where(~nan_mask)[0]
    if len(idx) == 0:
      return series 
    series[nan_mask] = np.interp(np.where(nan_mask)[0], idx, series[idx])
    return series

  x_interp = np.array([linear_interpolate(row) for row in x_raw])

  def three_sigma_clip(series):
    mean, std = np.mean(series) , np.std(series)
    upper = mean + 2 * std
    lower = mean - 2 * std
    return np.clip(series, lower, upper)

  x_clipped = np.array([three_sigma_clip(row) for row in x_interp])

  # Normalize 
  scaler = MinMaxScaler()
  x_norm = scaler.fit_transform(x_clipped.T).T

  # Removing the Data which more than 50% of the data is zero
  zero_ratio = (x_norm == 0).mean(axis=1)
  keep_idx = zero_ratio <= 0.5
  x_clean = x_norm[keep_idx]
  y_clean = y[keep_idx]

  print(f'Removed {np.sum(~keep_idx)} high-zero samples. Remaining:{x_clean.shape[0]}')
  return x_clean, y_clean




def extract_advanced_features(x):
  # Extracting 19 features
  n_samples = x.shape[0]
  features = []

  for i in range(n_samples):
    s = x[i]
    mean = np.mean(s)
    std = np.std(s)
    min_v = np.min(s)
    max_v = np.max(s)
    median = np.median(s)
    q25, q75 = np.percentile(s, [25, 75])
    iqr = q75 - q25

    grad = np.gradient(s)
    mean_grad = np.mean(grad)
    std_grad = np.std(grad)

    last_week_mean = np.mean(s[-7:])
    last_month_mean = np.mean(s[-30:]) if len(s) >= 30 else mean

    weeks = s[:147 *7].reshape(147, 7) if len(s) >= 147 * 7 else s[:len(s)//7 * 7].reshape(-1, 7)
    if weeks.shape[0] > 0:
      weekly_means = np.mean(weeks, axis = 1)
      mean_weekly = np.mean(weekly_means)
      std_weekly = np.std(weekly_means)
      trend_weekly = np.polyfit(range(len(weekly_means)), weekly_means, 1)[0]

    else:
      mean_weekly = mean
      std_weekly = 0
      trend_weekly = 0


    fft = np.fft.fft(s)
    fft_mag = np.abs(fft[:50])
    energy = np.sum(fft_mag ** 2)
    dominant_freq = np.argmax(fft_mag[1:]) + 1 if len(fft_mag) > 1 else 0

    cv = std / (mean + 1e-8)
    max_min_ratio = max_v / (min_v + 1e-8)

    feat = [ mean, std, min_v, max_v, median, q25, q75, iqr, mean_grad, 
            std_grad, last_week_mean, last_month_mean, mean_weekly, std_weekly, 
            trend_weekly, energy, dominant_freq, cv, max_min_ratio]
    
    features.append(feat)

  return np.array(features)
    

  
    
