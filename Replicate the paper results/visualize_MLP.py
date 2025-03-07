import networkx as nx
import matplotlib.pyplot as plt

def draw_alu_mlp():
    # Create a directed graph
    G = nx.DiGraph()

    # -----------------------------
    # 1) Add nodes
    # -----------------------------
    input_nodes = ["A0", "A1", "B0", "B1", "Op0", "Op1"]
    hidden_nodes = [f"H{i}" for i in range(1, 7)]  # 6 hidden neurons
    output_nodes = ["O0", "O1"]  # 2-bit output

    # Add them to the graph
    for n in input_nodes + hidden_nodes + output_nodes:
        G.add_node(n)

    # -----------------------------
    # 2) Add edges (fully connected from input->hidden->output)
    # -----------------------------
    for inp in input_nodes:
        for h in hidden_nodes:
            G.add_edge(inp, h)

    for h in hidden_nodes:
        for out in output_nodes:
            G.add_edge(h, out)

    # -----------------------------
    # 3) Draw the graph
    # -----------------------------
    # We'll position them in distinct "layers":
    pos = {}
    # Input layer (y around some range, x=0)
    for i, inp in enumerate(input_nodes):
        pos[inp] = (0, i)

    # Hidden layer (x=1)
    for i, h in enumerate(hidden_nodes):
        pos[h] = (1, i)

    # Output layer (x=2)
    for i, out in enumerate(output_nodes):
        pos[out] = (2, i)

    # Draw
    plt.figure()
    nx.draw(G, pos, with_labels=True, arrows=True)

    # Save the figure to a file
    plt.savefig("mlp_graph.png")

    # Show the figure (requires X-forwarding with MobaXterm if you want a pop-up)
    plt.show()

# Run the function to display the MLP graph
if __name__ == "__main__":
    draw_alu_mlp()
