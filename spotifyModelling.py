# Data Science Exploration

# This script will fit models on the Top Hits on Spotify from 2000-2019 
# - https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019/data 
# - This dataset contains audio statistics of the top 2000 tracks on Spotify from 2000-2019. The data contains about 18 columns each describing the track and it's qualities.

# Import required libaries 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Read in the data 
spotifyData = pd.read_csv('spotify.csv')

# Fit A GLM Model
## Will fit a GLM to the data to predict the year of the song

# Define variables, dropping all categorical features, 80-20 train/test split 
X = spotifyData.drop(columns=['year', 'artist', 'song', 'genre'])
y = spotifyData['year']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the GLM
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model performance 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Results:
# Mean Squared Error: 30.849836548281324
# R-squared: 0.14436197290701924
# These results show that the model is not very useful at predicting the year of the song 



# Fit A Random Forest Classifier 
# Will fit a Random Forest model to the data to predict the genre of the song 

# Select relevant features and target variable
X2 = spotifyData.drop(columns=['artist', 'song', 'genre'])
y2 = spotifyData['genre']

# Encode categorical target variable (artist)
label_encoder = LabelEncoder()
y2 = label_encoder.fit_transform(y2)

# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=41)

# Initialize and train the model (e.g., Random Forest Classifier)
model = RandomForestClassifier()
model.fit(X2_train, y2_train)

# Make predictions on the testing set
y2_pred = model.predict(X2_test)

# Evaluate the model using accuracy, precision and recall 
accuracy = accuracy_score(y2_test, y2_pred)
precision = precision_score(y2_test, y2_pred, average='weighted', zero_division=0)  # Weighted average precision
recall = recall_score(y2_test, y2_pred, average='weighted', zero_division=0)  # Weighted average recall

# Print the metrics 
print("Accuracy: ", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Input my own data to test the model 
example_input = {
    'duration_ms': 228700,
    'explicit': 1,  
    'year': 2003, 
    'popularity': 66,
    'danceability': 0.3,
    'energy': 0.6,
    'key': 3,
    'loudness': 3,
    'mode': 3.5,
    'speechiness': 0.9,
    'acousticness': 0.3, 
    'instrumentalness': 0.3, 
    'liveness': 0.2,
    'valence': 0.3,
    'tempo': 35
}

example_df = pd.DataFrame([example_input])
predicted_genre = model.predict(example_df)
predicted_genre_names = label_encoder.inverse_transform(predicted_genre)
print("Predicted Genre:", predicted_genre_names)

