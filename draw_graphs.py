import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

def draw_graph(graph_file, prediction_file, out_file):

    file_ = open(graph_file, "r")
    lines = file_.readlines()
    file_.close()

    # dont draw graph if graph is too big
    if len(lines) > 5000:
        print("graph of file", graph_file, "is too large to be drawn")
        return None

    prediction_df = pd.read_csv(prediction_file, index_col=0)

    dict_contig_length = {}

    for line in lines:
        if line.split()[0] == "S":
            dict_contig_length[int(line.split()[1])] = len(line.split()[2])

    tuple_node1_node2 = []

    for line in lines:
        if line.split()[0] == "L":
            tuple_node1_node2.append((int(line.split()[1]), int(line.split()[3])))

    # generate graph

    G = nx.Graph()

    for tpl in tuple_node1_node2:
        G.add_edge(tpl[0], tpl[1])

    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))
    # remove all isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))

    node_sizes = []
    node_colors = []
    plasmid_nodes = set()

    for node_ in nx.nodes(G):
        seq_length = dict_contig_length[node_]

        if seq_length <= 100:
            node_sizes.append(20)
        elif seq_length <= 1000:
            node_sizes.append(40)
        elif seq_length <= 10000:
            node_sizes.append(80)
        else:
            node_sizes.append(160)

        try:
            label = prediction_df.loc[node_, "predicted_label"]
            if label == "Chromosome":
                node_colors.append("blue")
            elif label == "Plasmid":
                node_colors.append("green")
                plasmid_nodes.add(node_)
            elif label == "Ambiguous":
                node_colors.append("black")
            else:
                node_colors.append("grey")
        except KeyError:
            node_colors.append("grey")

    plasmid_edges = []
    for edge in nx.edges(G):
        if edge[0] in plasmid_nodes or edge[1] in plasmid_nodes:
            plasmid_edges.append(edge)

    # draw graph

    fig, ax = plt.subplots(figsize=(15, 10))

    pos = nx.kamada_kawai_layout(G)
    nx.draw(
        G, pos=pos, node_size=node_sizes, node_color=node_colors, alpha=0.8, width=0.2
    )  # , with_labels = True

    # remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    nx.draw_networkx_edges(G, pos, edgelist=plasmid_edges, width=0.8)

    # custom legends
    legend1_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="<= 100 bp",
            markerfacecolor="grey",
            markersize=6,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="<= 1 kbp",
            markerfacecolor="grey",
            markersize=9,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="<= 10 kbp",
            markerfacecolor="grey",
            markersize=12,
            alpha=0.7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="> 10 kbp",
            markerfacecolor="grey",
            markersize=15,
            alpha=0.7,
        ),
    ]

    legend2_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="chromosome",
            markerfacecolor="blue",
            markersize=12,
            alpha=0.7,
        )
    ]
    if "black" in node_colors:
        legend2_elements += [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="ambiguous",
                markerfacecolor="black",
                markersize=12,
                alpha=0.7,
            )
        ]
    if "green" in node_colors:
        legend2_elements += [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="plasmid",
                markerfacecolor="green",
                markersize=12,
                alpha=0.7,
            )
        ]

    legend_1 = plt.legend(
        handles=legend1_elements,
        loc="upper left",
        bbox_to_anchor=(1, 0.9),
        title="lengths",
    )
    legend_2 = plt.legend(
        handles=legend2_elements,
        loc="upper left",
        bbox_to_anchor=(0.995, 0.71),
        title="label",
    )

    plt.gca().add_artist(legend_1)
    plt.gca().add_artist(legend_2)

    plt.title("Predicted Graph")

    plt.savefig(
        out_file, dpi=600, format="png", bbox_inches="tight"
    )
    plt.clf()
    plt.close()

    return None
