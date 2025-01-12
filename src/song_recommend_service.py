import logging

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import json
# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class SongRecommendService:
    def __init__(self, songs_csv, user_ratings_csv=None, num_epochs=100, hidden_dim=64, lr=0.001):
        # Load the song data
        self.songs_data = pd.read_csv(songs_csv)

        # If user ratings are provided (optional), load the data
        self.user_ratings_data = pd.read_csv(user_ratings_csv) if user_ratings_csv else None

        # Preprocess the song data
        self.features_encoded, self.song_ids = self.preprocess_data()

        # Initialize model parameters
        self.input_dim = self.features_encoded.shape[1]
        self.hidden_dim = hidden_dim
        self.output_dim = 1  # Predicted score for each song (e.g., rating)
        self.num_epochs = num_epochs
        self.lr = lr

        # Initialize the neural network model
        self.model = self.ContentBasedModel(self.input_dim, self.hidden_dim, self.output_dim)

        # Loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    class ContentBasedModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SongRecommendService.ContentBasedModel, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

        def get_state(self):
            """Returns the model's state dictionary (weights and biases)."""
            return self.state_dict()

        def set_state(self, state_dict):
            """Sets the model's state using a provided state dictionary."""
            self.load_state_dict(state_dict)

    def preprocess_data(self):
        """Preprocess the song data (encoding categorical features and scaling numerical ones)."""
        # Extract features (Assuming 'genre', 'artist', 'tempo', 'duration' are available in the dataset)
        features = self.songs_data[['genre', 'artist', 'tempo', 'duration']]

        # One-hot encode categorical features (genre, artist)
        features_encoded = pd.get_dummies(features, columns=['genre', 'artist'], drop_first=True)

        # Standardize numerical features (tempo, duration)
        scaler = StandardScaler()
        features_encoded[['tempo', 'duration']] = scaler.fit_transform(features_encoded[['tempo', 'duration']])

        # Convert any remaining object columns to numeric and handle missing values
        features_encoded = features_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Convert the data into a numpy array with the correct type
        features_encoded = features_encoded.astype('float32')

        # Get song ids for later use
        song_ids = self.songs_data['song_id'].values

        return features_encoded, song_ids

    def train_model(self):
        """Train the model over multiple epochs."""
        # Convert features to PyTorch tensor
        X = torch.tensor(self.features_encoded.values, dtype=torch.float32)

        # Dummy target ratings (you can replace with actual user ratings if available)
        # In a real scenario, you would use ratings from the 'user_ratings_data'
        target = torch.randn(X.shape[0])  # Random target ratings as placeholders

        # Train the model for the specified number of epochs
        for epoch in range(self.num_epochs):
            self.model.train()

            # Forward pass: Compute predicted ratings for all songs
            outputs = self.model(X).squeeze()

            # Compute the loss
            loss = self.criterion(outputs, target)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print the loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def get_song_recommendations(self, title, top_n=5):
        """Recommend the top N songs most similar to a given song."""
        self.model.eval()  # Set the model to evaluation mode

        # Get the index of the song based on song_id
        song_titlex = self.songs_data[self.songs_data['title'] == title].index[0]

        # Get the features of the given song
        song_features = torch.tensor(self.features_encoded.iloc[song_titlex].values, dtype=torch.float32).unsqueeze(0)

        # Predict the rating for the song
        with torch.no_grad():
            predicted_rating = self.model(song_features).item()

        # Get the predicted ratings for all songs
        all_song_features = torch.tensor(self.features_encoded.values, dtype=torch.float32)
        with torch.no_grad():
            all_song_predictions = self.model(all_song_features).squeeze()

        # Get the top N most similar songs based on predicted ratings
        sorted_indices = torch.argsort(all_song_predictions, descending=True)
        top_n_indices = sorted_indices[1:top_n + 1]  # Exclude the song itself

        # Get the song IDs of the top N recommended songs
        recommended_song_ids = self.songs_data.iloc[top_n_indices]['title'].values

        return recommended_song_ids

    def recommend_songs_for_user(self, user_id, top_n=5):
        """Recommend the top N songs for a user based on their previous interactions."""
        if self.user_ratings_data is None:
            raise ValueError("User ratings data is required for this function.")

        # Get the songs the user has rated
        user_songs = self.user_ratings_data[self.user_ratings_data['user_id'] == user_id]['song_id'].values

        # Initialize a list to store recommended songs
        recommended_songs = []

        # For each song rated by the user, find similar songs
        for song_id in user_songs:
            similar_songs = self.get_song_recommendations(song_id, top_n=top_n)
            recommended_songs.extend(similar_songs)

        # Remove duplicates and return top N unique songs
        recommended_songs = list(set(recommended_songs))

        return recommended_songs[:top_n]

    def extract_song_from_string(self, text):
        # Check each title against the provided string
        for title in self.songs_data['title']:
            if title.lower() in text.lower():
                logging.info("[USER REQUEST] Song: {}".format(title))
                return title
        return "Blinding Lights"


# Example usage:

# 1. Initialize the recommendation service
# recommendation_service = SongRecommendService(songs_csv='songs.csv', user_ratings_csv='user_ratings.csv')

# 2. Train the model
# recommendation_service.train_model()

# 3. Get song recommendations for a specific song
# recommended_songs = recommendation_service.get_song_recommendations("Bohemian Rhapsody", top_n=5)
# print(f"Top 5 recommended songs similar to song Bohemian Rhapsody: {recommended_songs}")
#
# recommended_songs = recommendation_service.get_song_recommendations("Smells Like Teen Spirit", top_n=5)
# print(f"Top 5 recommended songs similar to song Smells Like Teen Spirit: {recommended_songs}")

# recommended_songs = recommendation_service.get_song_recommendations("Hotel California", top_n=5)
# print(f"Top 5 recommended songs similar to song Hotel California: {recommended_songs}")

# 4. Recommend songs for a user
# user_recommendations = recommendation_service.recommend_songs_for_user(user_id=104, top_n=5)
# print(f"Top 5 recommended songs for user 104: {user_recommendations}")
