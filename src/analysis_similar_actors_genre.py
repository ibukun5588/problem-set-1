'''
PART 2: SIMILAR ACTROS BY GENRE
Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''



#Write your code below
   
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import DistanceMetric
from datetime import datetime

def build_genre_matrix(file_path):
    """
    Build a genre matrix from the JSON dataset.
    
    :param file_path: Path to the JSON file
    :return: DataFrame with actors, genres , and  a counts for the num of appearances
    """
    genre_dict = {}
    
    # Read the JSON file
    with open(file_path, 'r') as in_file:
        for line in in_file:
            # Load the movie from this line
            this_movie = json.loads(line)
            genres = this_movie['genres']
            
            # Update the genre count for each actor in the movie
            for actor_id, actor_name in this_movie['actors']:
                if actor_id not in genre_dict:
                    genre_dict[actor_id] = {'name': actor_name, 'genres': {genre: 0 for genre in genres}}
                for genre in genres:
                    if genre not in genre_dict[actor_id]['genres']:
                        genre_dict[actor_id]['genres'][genre] = 0
                    genre_dict[actor_id]['genres'][genre] += 1

    # Convert the genre dictionary to a DataFrame
    actors = []
    genres = set()
    for actor_id in genre_dict:
        actors.append(actor_id)
        genres.update(genre_dict[actor_id]['genres'].keys())
    
    genre_matrix = pd.DataFrame(index=actors, columns=sorted(genres)).fillna(0)
    
    for actor_id in genre_dict:
        for genre in genre_dict[actor_id]['genres']:
            genre_matrix.at[actor_id, genre] = genre_dict[actor_id]['genres'][genre]
    
    return genre_matrix, {actor_id: genre_dict[actor_id]['name'] for actor_id in genre_dict}

def find_similar_actors(genre_matrix, actor_id, actor_names, metric='cosine'):
    """
    Find the top 10 most similar actors to the given actor based on genre appearances.
    
    :param genre_matrix: DataFrame with actors as rows and genres as columns
    :param actor_id: ID of the query actor
    :param actor_names: Dictionary of actor IDs to names
    :param metric: Distance metric to use ('cosine' or 'euclidean')
    :return: DataFrame with the top 10 most similar actors
    """
    # Ensure the actor_id is in the genre matrix
    if actor_id not in genre_matrix.index:
        raise ValueError(f"Actor ID {actor_id} not found in the genre matrix")
    
    if metric == 'cosine':
        # Compute cosine distances
        distances = cosine_distances(genre_matrix)
    elif metric == 'euclidean':
        # Compute euclidean distances
        dist = DistanceMetric.get_metric(metric)
        distances = dist.pairwise(genre_matrix)
    else:
        raise ValueError(f"Unrecognized metric {metric}")
    
    # Get the distances for the query actor
    actor_idx = genre_matrix.index.get_loc(actor_id)
    actor_distances = distances[actor_idx]
    
    # Create a DataFrame with the distances and sort to find the top 10 most similar actors
    similar_actors = pd.DataFrame({
        'actor_id': genre_matrix.index,
        'actor_name': [actor_names[aid] for aid in genre_matrix.index],
        'distance': actor_distances
    }).sort_values(by='distance')
    
    # Exclude the query actor itself
    similar_actors = similar_actors[similar_actors['actor_id'] != actor_id]
    
    return similar_actors.head(10)

if __name__ == "__main__":
    file_path = 'C:/Users/Ibu G/Documents/414/problem-set-1/imdb_movies_2000to2022.prolific.json'
    
    # Build the genre matrix
    genre_matrix, actor_names = build_genre_matrix(file_path)
    
    # Query actor ID (dynamic input)
    query_actor_id = "nm1165110"  # Chris Hemsworth but it's dynamic as well
    query_actor_name = actor_names[query_actor_id]  # Get the query actor's name
    
    # Find the top 10 most similar actors using cosine distance
    similar_actors_cosine = find_similar_actors(genre_matrix, query_actor_id, actor_names, metric='cosine')
    
    # Output the result to a CSV file
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_cosine = f"C:/Users/Ibu G/Documents/414/problem-set-1/similar_actors_genre_{current_datetime}.csv"
    similar_actors_cosine.to_csv(output_path_cosine, index=False)
    
    print(f"Top 10 actors most similar to {query_actor_name} (Cosine Distance):")
    print(similar_actors_cosine)
    print(f"Results located at {output_path_cosine}")
    
    # Find the top 10 most similar actors using euclidean distance
    similar_actors_euclidean = find_similar_actors(genre_matrix, query_actor_id, actor_names, metric='euclidean')
    
    print(f"Top 10 actors most similar to {query_actor_name} (Euclidean Distance):")
    print(similar_actors_euclidean)
    
    # Describe how the list changes based on Euclidean distance
    print("Cosine distance focuses on proportional genre similarity, leading to varied results, while Euclidean distance emphasizes total genre appearances, resulting in different similar actors.")
