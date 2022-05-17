# plASgraph - contig labelling of bacterial assembly graphs

## Overview

Identification of plasmids and plasmid genes from sequencing data is an important question regarding antimicrobial resistance spread and other One-Health issues. We built a graph neural network (GNN), which incorporates sequence features as well as the assembly graph of the given input contigs. Our tool plASgraph takes the architecture of the assembly graph for the given input contigs directly into consideration as neighbouring contigs in the assembly graph are more likely to belong to the same class. We accurately predict long contigs and shorter contigs below 1 kb, which allows the identification of more complete plasmids from unresolved assemblies. PlASgraph is able to generalize across different species. This might allow the classification of plasmid contigs in previously unknown species for which no complete assembly sequence is present in any public data repository, yet.

<p align="center">
  <img src="/figures/model_architecture_figure_github.png" alt="drawing" width="600"/>
</p>


## Installation

~~~
git clone https://github.com/cchauve/plASgraph.git
~~~

#### Required Python modules:

All modules can be installed using pip.

  - NetworkX  2.6.3+
  - Pandas  1.3.5+
  - NumPy  1.21.5+
  - Scikit-learn  0.23.1+
  - Scipy 1.4.1+
  - Biopython  1.79+
  - Matplotlib  3.5.1+
  - TensorFlow  2.8.0+
  - Spektral  1.0.8+


## Usage

As input for plASgraph, only the assembly graph file (.gfa) is required. For all our analyses, we used the assembly graph output provided by Unicycler (Wick *et al*., 2017).

~~~

python3 plASgraph.py 

-i STR    full path to assembly graph file (.gfa) 


Optional:

-o STR   full path to output file (.csv)

--draw_graph

~~~

PlASgraph predicts each contig as either plasmidic, chromosomal or ambiguous. The output file contains four columns: I) the name of the contig, II) the plasmid score, III) the chromosome score and IV) the predicted label.


#### Usage example (test set):

~~~
python3 plASgraph.py -i example_data/c_freundii_assembly.gfa -o output.csv --draw_graph
~~~
