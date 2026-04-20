"""
Script to transform .txt graph files into formatted tables
"""
import pandas as pd
import os
import random

def get_df_from_graph_file(filepath):
    """Parse a graph file and optionally export to CSV"""
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    start = lines[0].strip().split()[1]
    goal = lines[0].strip().split()[2]
    
    edges = []
    for line in lines[1:]:
        if line.strip():  # Skip empty lines
            parts = line.strip().split()
            if len(parts) == 4:
                node1, node2, distance, fatigue = parts
                edges.append({
                    'Node 1': node1,
                    'Node 2': node2,
                    'Distance': distance,
                    'Fatigue': fatigue
                })
    
    # Create DataFrame
    df = pd.DataFrame(edges)
    
    return df, (start, goal)

def create_matching(df):
    """Create a matching from a CSV file"""
    
    combined_list = df['Node 1'].tolist() + df['Node 2'].tolist()
    unique_nodes = sorted(set(combined_list))  # Sort to make mapping deterministic!

    dict_matching : dict = {}
    cpt = 0
    for node in unique_nodes:
        if node not in dict_matching:
            dict_matching[node] = cpt
            cpt += 1

    return dict_matching

def modify_nodes_names(df, dict_matching):
    """Modify node names in the DataFrame based on the matching dictionary"""
    df['Node 1'] = df['Node 1'].map(dict_matching)
    df['Node 2'] = df['Node 2'].map(dict_matching)
    return df

def main(filepath, export_csv=True):
    """Main function to process the graph file and create matching"""
    df, (start, goal) = get_df_from_graph_file(filepath)
    dict_matching = create_matching(df)
    df = modify_nodes_names(df, dict_matching)
    start = dict_matching[start]
    goal = dict_matching[goal]

    if export_csv:
        filename = os.path.basename(filepath).replace('.txt', '.csv')
        output_csv_path = os.path.join('formated_examples', filename)
        os.makedirs('formated_examples', exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        print(f"Data exported to {output_csv_path}")
    
    return dict_matching, df, (start, goal)

def load_missions(filepath: str) -> list[tuple[int, int]]:
    """Read missions from a file where each line contains 'start_node end_node'."""
    missions = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                missions.append((int(parts[0]), int(parts[1])))
        print(f"Loaded {len(missions)} missions from {filepath}")
    except Exception as e:
        print(f"Error loading missions from {filepath}: {e}")
    return missions


def generate_missions(num_missions: int, max_node: int, output_filepath: str) -> None:
    """Generate random missions and save to output_filepath."""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, 'w') as f:
        for _ in range(num_missions):
            start = random.randint(0, max_node)
            end = random.randint(0, max_node)
            while end == start:
                end = random.randint(0, max_node)
            f.write(f"{start} {end}\n")


def generate_all_mission_files() -> None:
    """Generate all 6 mission files in the example_missions/ folder."""
    folder = 'example_missions'
    os.makedirs(folder, exist_ok=True)

    configs = [
        ('medium-smallmission.txt',  5,  99),
        ('medium-mediummission.txt', 20,  99),
        ('medium-largemission.txt',  50,  99),
        ('large-smallmission.txt',    5, 9999),
        ('large-mediummission.txt',  20, 9999),
        ('large-largemission.txt',   50, 9999),
    ]

    print('=' * 40)
    for filename, num_missions, max_node in configs:
        filepath = os.path.join(folder, filename)
        generate_missions(num_missions, max_node, filepath)
        print(f"Generated {filepath}: {num_missions} missions, nodes 0-{max_node}")
    print('=' * 40)


if __name__ == '__main__':
    generate_all_mission_files()