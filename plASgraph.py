import itertools
import pandas as pd
import networkx as nx
import numpy as np

from draw_graphs import draw_graph

from Bio.Seq import Seq

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras

from spektral.data import Dataset
from spektral.data import Graph
from spektral.transforms import GCNFilter
from spektral.data.loaders import SingleLoader

from argparse import ArgumentParser

input_ = ArgumentParser()
input_.add_argument("-i", dest = "graph_file", required = True)
input_.add_argument("-o", dest = "output_file", default = "prediction.csv")
input_.add_argument('--draw_graph', dest = 'draw_graph', default = False, action = 'store_true')
args = input_.parse_args()

print("----- loading model -----")
model = keras.models.load_model("model_plasgraph_generalized")


class Networkx_to_Spektral(Dataset):
    def __init__(self, nx_graph, **kwargs):
        self.nx_graph = nx_graph

        super().__init__(**kwargs)

    def read(self):

        x = np.array(
            [self.nx_graph.nodes[node_name]["x"] for node_name in self.nx_graph.nodes]
        )

        y = np.array(
            [self.nx_graph.nodes[node_name]["y"] for node_name in self.nx_graph.nodes]
        )

        a = nx.adjacency_matrix(self.nx_graph)
        a.setdiag(0)
        a.eliminate_zeros()

        # We must return a list of Graph objects
        return [Graph(x=x.astype(float), a=a.astype(float), y=y.astype(float))]

# function for pentamer distributions

kmer_length = 5

k_mers = ["".join(x) for x in itertools.product("ACGT", repeat=kmer_length)]


fwd_kmers = []
rev_kmers = []

for k_mer in k_mers:
    if not ((k_mer in fwd_kmers) or (k_mer in rev_kmers)):
        fwd_kmers.append(k_mer)
        rev_kmers.append(str(Seq(k_mer).reverse_complement()))


def get_kmer_distribution(
    sequence, k_mers=k_mers, fwd_kmers=fwd_kmers, kmer_length=5, scale=False
):
    if len(sequence) < 5:
        return [0] * int(4**kmer_length / 2)
    dict_kmer_count = {}

    for k_mer in k_mers:
        dict_kmer_count[k_mer] = 0

    for i in range(len(sequence) + 1 - kmer_length):
        kmer = sequence[i : i + kmer_length]
        try:
            dict_kmer_count[kmer] += 1
        except KeyError:
            pass

    k_mer_counts = [
        dict_kmer_count[k_mer] + dict_kmer_count[str(Seq(k_mer).reverse_complement())]
        for k_mer in fwd_kmers
    ]

    if scale:
        scaler = MinMaxScaler()
        k_mer_counts = scaler.fit_transform(np.array(k_mer_counts).reshape(-1, 1))
        k_mer_counts = list(k_mer_counts.flatten())

    return k_mer_counts


# function for  GC content


def get_gc_content(seq):
    number_gc = 0
    number_acgt = 0
    for base in seq.lower():
        if base in "gc":
            number_gc += 1
        if base in "acgt":
            number_acgt += 1
    try:
        gc_content = round(number_gc / number_acgt, 4)
    except ZeroDivisionError:
        gc_content = 0.5
    return gc_content


print("----- extracting features -----")

dict_contig_length = {}
dict_contig_length_normalized = {}
tuple_node1_node2 = []
dict_contig_gc = {}
dict_contig_kmer = {}
dict_contig_coverage = {}
dict_contig_label = {}

dict_contig_kmer_euclidean_distance = {}
dict_contig_num_edges = {}

file_ = open(args.graph_file, "r")
lines = file_.readlines()
file_.close()

# get gc of whole seq

whole_seq = ""

for line in lines:
    if line.split()[0] == "S":
        whole_seq += line.strip().split()[2]

gc_of_whole_seq = get_gc_content(whole_seq)

# get contig lengths and max length

max_contig_length = 0

for line in lines:
    if line.split()[0] == "S":
        dict_contig_length[int(line.split()[1])] = len(line.split()[2])
        if len(line.split()[2]) > max_contig_length:
            max_contig_length = len(line.split()[2])

# get normalized contig lengths and max length

for line in lines:
    if line.split()[0] == "S":
        dict_contig_length_normalized[int(line.split()[1])] = (
            len(line.split()[2]) / max_contig_length
        )


# get graph edges

for line in lines:
    if line.split()[0] == "L":
        tuple_node1_node2.append((int(line.split()[1]), int(line.split()[3])))

# get gc content

for line in lines:
    if line.split()[0] == "S":
        dict_contig_gc[int(line.split()[1])] = (
            get_gc_content(line.split()[2]) - gc_of_whole_seq
        )

# get pentamer distributions

for line in lines:
    if line.split()[0] == "S":
        dict_contig_kmer[int(line.split()[1])] = get_kmer_distribution(
            line.split()[2], k_mers=k_mers, scale=True
        )

# get euclidian distance of pentamer distribution for each node

# generate dict with all contigs of current isolate and their pentamer distribution
dict_contig_kmer_current_isolate = {}

for line in lines:
    if line.split()[0] == "S":
        dict_contig_kmer_current_isolate[int(line.split()[1])] = get_kmer_distribution(
            line.split()[2], k_mers=k_mers
        )

# calculate total pentamer distribution and scale between 0 and 1
all_kmer_counts = [
    sum(x) for x in zip(*list(dict_contig_kmer_current_isolate.values()))
]
scaler = MinMaxScaler()
all_kmer_counts = scaler.fit_transform(np.array(all_kmer_counts).reshape(-1, 1))
all_kmer_counts = list(all_kmer_counts.flatten())

# get euclidean distance for each contig and add to dict
for contig in dict_contig_kmer_current_isolate:
    kmer_distribution = np.array(dict_contig_kmer_current_isolate[contig])
    scaler = MinMaxScaler()
    scaled_kmer_distribution = scaler.fit_transform(
        np.array(kmer_distribution).reshape(-1, 1)
    )
    scaled_kmer_distribution = list(scaled_kmer_distribution.flatten())
    dict_contig_kmer_euclidean_distance[contig] = np.linalg.norm(
        np.array(all_kmer_counts) - np.array(scaled_kmer_distribution)
    )

# get coverage

for line in lines:
    if line.split()[0] == "S":
        assert "dp:f" in line, "no scaled depth of coverage in graph file"
        split_line = line.strip().split()
        for element in split_line:
            if "dp:f" in element:
                dict_contig_coverage[int(line.split()[1])] = round(
                    float(element.split(":")[-1]), 2
                )

# generate networkx graph

G = nx.Graph()

for tpl in tuple_node1_node2:
    G.add_edge(tpl[0], tpl[1])

# get number of edges per contig

for contig_ in G.nodes:
    dict_contig_num_edges[contig_] = len(list(G.neighbors(contig_)))

# make feature dict

dict_contig_list_coverage_gc_kmer = {}

for contig_ in G.nodes:
    dict_contig_list_coverage_gc_kmer[contig_] = [
        dict_contig_coverage[contig_],
        dict_contig_gc[contig_],
        dict_contig_kmer_euclidean_distance[contig_],
        dict_contig_num_edges[contig_],
        dict_contig_length_normalized[contig_],
    ]

# add features to graph nodes
nx.set_node_attributes(G, dict_contig_list_coverage_gc_kmer, "x")

# remove all nodes < 100 bp
for node in list(G.nodes):
    # print(node, "neighbors:", list(G.neighbors(node)), "len:", len(list(G.neighbors(node))))
    if dict_contig_length[node] < 100:
        if len(list(G.neighbors(node))):
            # print("connecting neighbors of node", node)
            neighbors = list(G.neighbors(node))
            all_new_edges = list(itertools.combinations(neighbors, 2))
            for edge in all_new_edges:
                G.add_edge(edge[0], edge[1])
        # print("removing node", node)
        G.remove_node(node)


# add empty labels to graph nodes
dict_contig_label = {}
for contig_ in dict_contig_coverage:
    dict_contig_label[contig_] = [0, 0]

nx.set_node_attributes(G, dict_contig_label, "y")

# generate spektral graph and predict labels
the_graph = Networkx_to_Spektral(G)

the_graph.apply(GCNFilter())

loader = SingleLoader(the_graph)

print("----- predicting contig labels -----")

preds = model.predict(loader.load(), steps=loader.steps_per_epoch)

# prediction to df
list_of_lists_with_prediction = []
for index, contig in enumerate(G.nodes):
    contig_name = contig
    plasmid_score = preds[index][0]
    chromosome_score = preds[index][1]
    # label
    if list(np.around(preds[index])) == [0, 0]:
        label = "not_labelled"
    elif list(np.around(preds[index])) == [0, 1]:
        label = "Chromosome"
    elif list(np.around(preds[index])) == [1, 0]:
        label = "Plasmid"
    elif list(np.around(preds[index])) == [1, 1]:
        label = "Ambiguous"
    list_of_lists_with_prediction.append(
        [contig_name, plasmid_score, chromosome_score, label]
    )

prediction_df = pd.DataFrame(
    list_of_lists_with_prediction,
    columns=[
        "contig_name",
        "plasmid_score",
        "chromosome_score",
        "predicted_label",
    ],
)

prediction_df.to_csv(args.output_file, index=False)

if args.draw_graph:
    print("----- drawing predicted graph -----")
    draw_graph(args.graph_file, args.output_file, args.output_file[0:-4] + "_graph.png")

print("----- done -----")