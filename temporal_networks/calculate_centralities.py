"""
Temporal Network Analysis: Centrality Measures Module

This module provides functions for computing various centrality measures that
quantify the importance and influence of individual nodes within networks.
Supports computation across temporal networks with proper handling of node labels.
"""

import pandas as pd
from typing import List, Optional


def calculate_centralities(graphs: List,
                          graph_labels: Optional[List[str]] = None,
                          filename: Optional[str] = "centralities_results.csv") -> pd.DataFrame:
    """
    Compute comprehensive centrality measures for nodes across multiple graphs.
    
    Calculates various centrality metrics for each node in a collection of networks,
    including degree, closeness, betweenness, eigenvector, PageRank, harmonic,
    eccentricity, clustering coefficient, and HITS scores.
    
    Parameters
    ----------
    graphs : list
        List of igraph.Graph objects to analyze
    graph_labels : list of str, optional
        Labels for each graph (e.g., ["2019-01", "2019-02", ...])
        If not provided, defaults to "Graph 1", "Graph 2", etc.
    filename : str, optional
        CSV filename for saving results (default: "centralities_results.csv")
        If None, results are not saved to file
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with centrality measures, one row per node per graph.
        Columns include: Graph, Node, and various centrality measures
        
    Examples
    --------
    >>> import igraph as ig
    >>> from temporal_networks import calculate_centralities
    >>> G1 = ig.Graph.Barabasi(n=50, m=2)
    >>> G2 = ig.Graph.Barabasi(n=50, m=2)
    >>> graphs = [G1, G2]
    >>> labels = ["2019-01", "2019-02"]
    >>> centralities = calculate_centralities(graphs, graph_labels=labels)
    >>> centralities.head()
    
    Notes
    -----
    Centrality measures computed:
    - Degree: Number of direct connections
    - Closeness: Average distance to all other nodes
    - Betweenness: Number of shortest paths node lies on
    - Eigenvector: Importance based on connections to important nodes
    - PageRank: Probabilistic measure of node importance
    - Harmonic: Closeness variant less sensitive to disconnected components
    - Eccentricity: Greatest distance to any other node
    - Clustering Coefficient: Tendency of neighbors to be connected
    - HITS Authority/Hub: Measures from hyperlink-induced topic search
    """
    
    # Validate inputs
    if not graphs:
        raise ValueError("graphs list cannot be empty")
    
    # Set up graph labels
    if graph_labels is None:
        graph_labels = [f"Graph {i+1}" for i in range(len(graphs))]
    elif len(graph_labels) != len(graphs):
        raise ValueError(f"graph_labels length ({len(graph_labels)}) must match graphs length ({len(graphs)})")
    
    # Initialize list to store centrality measures for all nodes across all graphs
    centralities_list = []

    # Iterate over each graph
    for graph_idx, graph in enumerate(graphs):
        graph_name = graph_labels[graph_idx]
        
        # Get node labels from graph attributes
        if "name" in graph.vs.attributes():
            node_labels = graph.vs["name"]
        elif "label" in graph.vs.attributes():
            node_labels = graph.vs["label"]
        else:
            # Default to node indices as strings
            node_labels = [f"Node_{i}" for i in range(graph.vcount())]
            print(f"Warning: Graph {graph_name} has no 'name' or 'label' attribute. Using node indices.")
        
        # Compute centrality measures
        try:
            degree_centrality = graph.degree()
        except:
            degree_centrality = [None] * graph.vcount()
        
        try:
            closeness_centrality = graph.closeness()
        except:
            closeness_centrality = [None] * graph.vcount()
        
        try:
            betweenness_centrality = graph.betweenness(directed=graph.is_directed())
        except:
            betweenness_centrality = [None] * graph.vcount()
        
        try:
            eigenvector_centrality = graph.eigenvector_centrality()
        except:
            eigenvector_centrality = [None] * graph.vcount()
        
        try:
            pagerank = graph.pagerank()
        except:
            pagerank = [None] * graph.vcount()
        
        try:
            harmonic_centrality = graph.harmonic_centrality()
        except:
            harmonic_centrality = [None] * graph.vcount()
        
        try:
            eccentricity = graph.eccentricity()
        except:
            eccentricity = [None] * graph.vcount()
        
        try:
            clustering_coefficient = graph.transitivity_local_undirected()
        except:
            clustering_coefficient = [None] * graph.vcount()
        
        try:
            authority_score = graph.authority_score()
        except:
            authority_score = [None] * graph.vcount()
        
        try:
            hub_score = graph.hub_score()
        except:
            hub_score = [None] * graph.vcount()
        
        # For each node, store all centrality measures
        for node_idx, node_label in enumerate(node_labels):
            centrality_dict = {
                "Graph": graph_name,
                "Node": node_label,
                "Node_Index": node_idx,
                "Degree_Centrality": degree_centrality[node_idx],
                "Closeness_Centrality": closeness_centrality[node_idx],
                "Betweenness_Centrality": betweenness_centrality[node_idx],
                "Eigenvector_Centrality": eigenvector_centrality[node_idx],
                "PageRank": pagerank[node_idx],
                "Harmonic_Centrality": harmonic_centrality[node_idx],
                "Eccentricity": eccentricity[node_idx],
                "Clustering_Coefficient": clustering_coefficient[node_idx],
                "HITS_Authority": authority_score[node_idx],
                "HITS_Hub": hub_score[node_idx],
            }
            centralities_list.append(centrality_dict)

    # Convert to DataFrame
    centralities_df = pd.DataFrame(centralities_list)

    # Save to CSV if requested
    if filename:
        try:
            centralities_df.to_csv(filename, index=False)
            print(f"✓ Centralities results saved to {filename}")
        except Exception as e:
            print(f"Error saving centralities to CSV: {e}")

    return centralities_df
