import networkx as nx
import matplotlib.pyplot as plt
import json
import os


def generate_small_world_network(n, k, p):
    """
    Generate a small-world network.

    Parameters:
    n (int): Number of nodes in the network.
    k (int): Number of neighbors each node is connected to.
    p (float): Probability of rewiring an edge.

    Returns:
    G (networkx.Graph): Generated small-world network.
    """
    G = nx.watts_strogatz_graph(n, k, p)
    mapping = {old: old + 1 for old in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G


def save_connections_to_json(G, agent_data_filename, output_filename):
    """
    Save the connection relationships and agent data to a JSON file.

    Parameters:
    G (networkx.Graph): Network graph.
    agent_data_filename (str): File name containing agent data.
    output_filename (str): Output file name.
    """
    with open(agent_data_filename, "r") as f:
        agent_data = json.load(f)

    for node in G.nodes():
        if node - 1 < len(agent_data):
            agent_data[node]["Connections"] = list(G.neighbors(node))

    with open(output_filename, "w") as f:
        json.dump(agent_data, f, indent=4)
    print(f"Agent data has been saved to {output_filename}")


def save_small_world_info(G, output_filename):
    """
    Save the basic information of the small-world network (total edges, total nodes, and connection relationships) to a JSON file.

    Parameters:
    G (networkx.Graph): Network graph.
    output_filename (str): Output file name.
    """
    small_world_info = {
        "Number_of_nodes": G.number_of_nodes(),
        "Number_of_edges": G.number_of_edges(),
        "Average_clustering_coefficient": nx.average_clustering(G),
        "Average_shortest_path_length": nx.average_shortest_path_length(G),
        "Connections": {node: list(G.neighbors(node)) for node in G.nodes()},
    }

    with open(output_filename, "w") as f:
        json.dump(small_world_info, f, indent=4)
    print(f"Small-world network information has been saved to {output_filename}")


# Parameter settings
n = 5  # Number of nodes
k = 3  # Number of neighbors each node is connected to
p = 0.2  # Probability of rewiring an edge

folder_name = str(n)
input_json = f"agent_data_{n}.json"
if not os.path.exists(folder_name):
    print(f"Folder {folder_name} does not exist.")
else:
    os.chdir(folder_name)
    if not os.path.exists(input_json):
        print(f"{input_json} does not exist.")
    else:
        output_json = f"agent_data_full_{n}_{k}_{p}.json"
        output_image = f"small_world_network_{n}_{k}_{p}.png"
        output_image_improved = f"small_world_network_improved_{n}_{k}_{p}.png"
        output_json_swn = f"small_world_info_{n}_{k}_{p}.json"
        output_files = [
            output_json,
            output_image,
            output_image_improved,
            output_json_swn,
        ]
        existing_files = [file for file in output_files if os.path.exists(file)]
        if existing_files:
            print("The following output files already exist:")
            for file in existing_files:
                print(f"- {file}")
        else:
            G = generate_small_world_network(n, k, p)
            pos = nx.circular_layout(G)
            nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray")
            plt.savefig(output_image, dpi=300, bbox_inches="tight")
            print(f"Network graph has been saved as {output_image}")
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(10, 8))
            nx.draw(
                G,
                pos,
                with_labels=True,
                node_size=500,
                node_color="lightblue",
                edge_color="gray",
                width=2,
                alpha=0.7,
                font_size=12,
                font_weight="bold",
            )
            plt.savefig(output_image_improved, dpi=300, bbox_inches="tight")
            print(f"Network graph has been saved as {output_image_improved}")
            # save_connections_to_json(G, input_json, output_json)
            save_small_world_info(G, output_json_swn)
