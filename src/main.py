'''
Pull down the imbd_movies dataset here and save to /data as imdb_movies_2000to2022.prolific.json
You will run this project from main.py, so need to set things up accordingly
'''

import json
import os
import requests
from datetime import datetime
import analysis_network_centrality
import analysis_similar_actors_genre

# Ingest and save the IMDb movies dataset
def download(url, save_path):
    getFile = requests.get(url)
    if getFile.status_code == 200:
        with open(save_path, 'w') as file:
            file.write(getFile.text)
        print(f"The Dataset is downloaded and  is saved at {save_path}")
    else:
        print("Failed to download the dataset")

# Call functions / instantiate objects from the two analysis .py files
def main():
    # Define file paths
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    json_file_path = os.path.join(data_dir, "imdb_movies_2000to2022.prolific.json")
    
    # Download dataset
    url = "https://github.com/cbuntain/umd.inst414/blob/main/data/imdb_movies_2000to2022.prolific.json?raw=true"
    download(url, json_file_path)
    
    # Part 1: Network Centrality Analysis
    print("\nRunning Network Centrality Analysis............")
    g = analysis_network_centrality.build_graph_from_json(json_file_path)
    centrality_df = analysis_network_centrality.compute_centrality_metrics(g)
    
    # Output the result to a CSV file
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path_centrality = os.path.join(data_dir, f"network_centrality_{current_datetime}.csv")
    centrality_df.to_csv(output_path_centrality, index=False)
    
    print(f"Centrality results are saved to {output_path_centrality}")
    
    # Part 2: Similar Actors by Genre
    print("\nRunning Similarity Analysis.........")
    genre_matrix, actor_names = analysis_similar_actors_genre.build_genre_matrix(json_file_path)
    
    # Query actor ID (dynamic input)
    query_actor_id = "nm1165110"  
    query_actor_name = actor_names[query_actor_id]
    
    # Find the top 10 most similar actors using cosine distance
    similar_actors_cosine = analysis_similar_actors_genre.find_similar_actors(genre_matrix, query_actor_id, actor_names, metric='cosine')
    
    # Output the result to a CSV file
    output_path_cosine = os.path.join(data_dir, f"similar_actors_genre_{current_datetime}.csv")
    similar_actors_cosine.to_csv(output_path_cosine, index=False)
    
    print(f"Top 10 actors most similar to {query_actor_name} (Cosine Distance):")
    print(similar_actors_cosine)
    print(f"Results located at {output_path_cosine}")
    
    # Find the top 10 most similar actors using euclidean distance
    similar_actors_euclidean = analysis_similar_actors_genre.find_similar_actors(genre_matrix, query_actor_id, actor_names, metric='euclidean')
    
    print(f"Top 10 actors most similar to {query_actor_name} (Euclidean Distance):")
    print(similar_actors_euclidean)
    
    # Describe how the list changes based on Euclidean distance
    print("Cosine distance focuses on proportional genre similarity, leading to varied results, while Euclidean distance emphasizes total genre appearances, resulting in different similar actors.")

if __name__ == "__main__":
    main()
