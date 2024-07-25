'''
PART 1: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Build a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to. 
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is line with the standards we're using in this class 
'''

import numpy as np
import pandas as pd
import networkx as nx
import json
import pandas as pd


def build_graph_from_json(file_path):

    """
    Build a graph from the JSON dataset.
    
    :param file_path: Path to the JSON file
    :return: A NetworkX graph
    """

    # Builds the graph
    g = nx.Graph()

    # Set up your dataframe(s) -> the df that's output to a CSV should include at least the columns 'left_actor_name', '<->', 'right_actor_name'
    with open(file_path, 'r') as in_file:
        # Don't forget to comment your code
        for line in in_file:
            # Load the movie from this line
            this_movie = json.loads(line)

            # Create a node for every actor
            for actor_id, actor_name in this_movie['actors']:
                g.add_node(actor_id, name=actor_name, ratings=[], votes=0)
            # Iterate through the list of actors, generating all pairs
            ## Starting with the first actor in the list, generate pairs with all subsequent actors
            ## then continue to second actor in the list and repeat
            i = 0 # counter
            for left_actor_id, left_actor_name in this_movie['actors']:
                for right_actor_id, right_actor_name in this_movie['actors'][i+1:]:
                    # Get the current weight, if it exists
                    if g.has_edge(left_actor_id, right_actor_id):
                        g[left_actor_id][right_actor_id]['weight'] += 1
                    else:
                        # Add an edge for these actors
                        g.add_edge(left_actor_id, right_actor_id, weight=1)
                i += 1
    
    return g


def compute_centrality_metrics(graph):
    """
    For finding centrality metrics for the graph.
    
    :param graph: A NetworkX graph
    :return: DataFrame with centrality metrics
    """
    # Compute degree centrality for the graph
    degree_centrality = nx.degree_centrality(graph)

    # Sort the centrality metrics in descending order
    sorted_centrality = sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True)
    
    # Prepare data for DataFrame
    data = {
        'actor_id': [],
        'actor_name': [],
        'degree_centrality': [],
    }
    
    for actor_id, centrality in sorted_centrality:
        data['actor_id'].append(actor_id)
        data['actor_name'].append(graph.nodes[actor_id]['name'])
        data['degree_centrality'].append(degree_centrality[actor_id])
       

    return pd.DataFrame(data)


# Print the info below
if __name__ == "__main__":
    file_path = 'C:/Users/Ibu G/Documents/414/problem-set-1/imdb_movies_2000to2022.prolific.json' 
    # Build the graph
    g = build_graph_from_json(file_path)
    
    # Compute centrality metrics
    centrality_df = compute_centrality_metrics(g)
    
    # Print basic graph information
    print(f" Total Nodes: {len(g.nodes)}")
    
    # Print the 10 most central nodes
    print(" The Top 10 Most Central Nodes:")
    print(centrality_df.head(10))
    
    # Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`
    from datetime import datetime
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"C:/Users/Ibu G/Documents/414/problem-set-1/network_centrality_{current_datetime}.csv"
    centrality_df.to_csv(output_path, index=False)
    
    print(f"The CSV is located at: {output_path}")