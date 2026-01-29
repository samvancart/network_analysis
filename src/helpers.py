import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def _prob_independent(overlap: int, pp: float) -> float:
    """
    Calculate the probability of an edge given the neighborhood overlap (i.e., common neighbors) between two nodes.

    This function computes the probability that at least one of the the common neighbors of 
    two nodes succeeds in creating the corresponding edge.

    Note:
        The overlap is incremented by 1 to account for the random edge selection in step 0.
    """
    overlap = overlap + 1  # adding 1 due to the random edge selection of step 0
    
    # Handle edge case where pp = 1.0
    if pp >= 1.0:
        return 1.0 if overlap > 0 else 0.0
    
    return -np.expm1(overlap * np.log1p(-pp)) # a numerically better way to say `1 - (1 - pp)**overlap`

    
def prob_to_add_now(new_overlap: int, prev_overlap: int, prob_function, **parameters) -> float:
    """
    Calculate the probability of adding a new connection based on the overlap.

    This function computes the probability of adding a new connection given the 
    new overlap and the previous overlap using a specified probability function.

    One can use this function to experiment with probability functions other than `prob_func`.

    Args:
        new_overlap (int): The new overlap value.
        prev_overlap (int): The previous overlap value.
        prob_function (callable): The probability function to use. 
        **parameters: Additional parameters to pass to the probability function.

    Returns:
        float: The probability of adding the new connection.
    """
    current = prob_function(new_overlap, **parameters)
    previous = prob_function(prev_overlap, **parameters)

    # Handle the case where previous = 1.0 (which would cause division by zero)
    if previous >= 1.0:
        return 1.0 if current > previous else 0.0
    
    return (current - previous)  / (1.0 - previous)


def find_largest_gap(results):
    grouped = results.groupby("p")
    mean_density = grouped["density"].mean()
    # order mean_density by density
    mean_density = mean_density.sort_values()
    mean_pp = mean_density.index

    mean_density = mean_density.values
    mean_pp = mean_pp.values

    # find the index of the largest density consecutive gap
    largest_gap = 0
    largest_gap_index = 0
    for i in range(len(mean_density)-1):
        gap = mean_density[i+1] - mean_density[i]
        if gap > largest_gap:
            largest_gap = gap
            largest_gap_index = i
    print(f"The largest gap in density is {largest_gap:.2f} between p={mean_pp[largest_gap_index]:.10f} and p={mean_pp[largest_gap_index+1]:.10f}")
    return largest_gap, mean_pp[largest_gap_index], mean_pp[largest_gap_index+1]

def extract_overlaps(G, **parameters):
    overlap_entries = []
    UU = list(G.nodes())
    for i, u in enumerate(UU):
        for v in UU[i+1:]:
            overlap = len(set(G.neighbors(u)) & set(G.neighbors(v)))
            edge_exists = int(G.has_edge(u, v))
            entry = {"G": id(G), "u": u, "v": v, "overlap": overlap, "edge_exists": edge_exists, 
                     **parameters}  
            
            overlap_entries.append(entry)
    return overlap_entries

def check_data_completeness(G, overlaps_data):
    """
    Check if we have data for all possible node pairs.
    
    Args:
        G: The graph
        overlaps_data: DataFrame with overlap data for this graph
    
    Returns:
        dict with statistics about data completeness
    """
    total_possible_pairs = G.number_of_nodes() * (G.number_of_nodes() - 1) // 2
    recorded_pairs = len(overlaps_data)
    
    # Count pairs by overlap value
    overlap_counts = overlaps_data['overlap'].value_counts().sort_index()
    
    # Count edges vs non-edges by overlap
    edge_stats = overlaps_data.groupby('overlap')['edge_exists'].agg(['count', 'sum', 'mean'])
    edge_stats.columns = ['total_pairs', 'edges', 'frequency']
    edge_stats['non_edges'] = edge_stats['total_pairs'] - edge_stats['edges']
    
    return {
        'total_possible_pairs': total_possible_pairs,
        'recorded_pairs': recorded_pairs,
        'missing_pairs': total_possible_pairs - recorded_pairs,
        'edge_stats_by_overlap': edge_stats
    }