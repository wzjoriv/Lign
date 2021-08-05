# Lign

[![Lign logo][logo-url]][repo-url]

A graph framework that can be used to implement graph convolutional networks (GCNs), geometry machine learning, continual lifelong learning on graphs and other graph-based machine learning methods alongside [PyTorch](https://pytorch.org)

[View Docs][docs-url] · [View Examples][examples-url] · [Report Bugs][bugs-url] · [Request Feature][bugs-url]

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Mozilla License][license-shield]][license-url]

## Table of Contents

- [Lign](#lign)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Future](#future)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## About The Project

Lign (Lifelong learning Induced by Graph Neural networks) is a framework that can be used for a wide range of graph-related problems including but not limited to graph convolutional networks (GCNs), geometry machine learning, continual lifelong learning on graphs and other graph-related machine learning methods with a tight integration to [PyTorch](https://pytorch.org). It was build as a component of my master thesis for a proposed lifelong learning techinique (also named Lign) that works on both graph and vector data.

Ligns currently supports a wide range of functionalities such as clustering techiniques, GCN PyTorch module, graph creation and processing, data set to graph convention, rehersal methods/retraining for GCNS and coventional neural networks, and more. Future planned additions include STL file to graph conversion, graph to STL model along with others features.

## Getting Started

These are instructions for getting started with development. If only interested in the package, please install Lign via ``pip install lign``, ``conda install lign -c josuecom -c pytorch`` or by downloading the [package][release-url] from github and saving it in the ``site-packages/`` directory of your python interpreter.

_For more details, please read the [developer documentation][dev-docs-url]_

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/JosueCom/Lign.git
   ```

2. Switch directory

   ```sh
   cd Lign
   ```

3. Install prerequisites

   _It is recommended to install [PyTorch](https://pytorch.org) via the official site first for optimal performance_

   ```sh
   pip install -r docs/dev/requirements.txt
   ```

   ```sh
   conda env create -f docs/dev/environment.yml -n lign
   conda activate lign
   ```

4. Install package

   ```sh
   pip install . --upgrade
   ```

## Usage

It is recommended to run the following instructions in a python console to view all produced output  

- Create a new graph, assign some values and view properties

   ```python
   import lign as lg
   import torch as th

   n = 5
   g = lg.Graph()
   g.add(n)
   g['x'] = th.rand(n, 3) ## Or, g.set_data('x', th.rand(n, 3))

   print(g)
   print(g['x']) ## Or, g.get_data('x')
   print(g[0]) ## Or, g.get_nodes(0)
   print(g[[1, 2]]) ## Or, g.get_nodes([1, 2])
   print(g[3:]) ## Or, g.get_nodes(slice(3, None))
   print(g[(4,)]) ## Or, g.get_edges(4)
   ```

- Process data with a conventional neural network

   ```python
   import lign as lg
   import torch as th
   from torch import nn

   n = 5
   g = lg.Graph().add(n, self_loop=False, inplace=True) ## No self loop edges added since no relational data is present
   g['x'] = th.rand(n, 3)
   print(g['x'])

   # Data that is not relational maybe be process without the need of ligh graphs
   x = = th.rand(n, 3)
   x = linear(x)
   print(x)

   # However, you can use if you desire to use graph strctures
   linear = nn.Linear(3, 2)
   g['x'] = linear(g['x']) ## Or, g.apply(linear, data = 'x')
   print(g['x'])
   ```

- Process relational data with a GCN

   ```python
   import lign as lg
   import torch as th
   from torch import nn
   from lign.nn import GCN
   from lign.utils.functions import sum_tensors

   n = 5
   g = lg.Graph().add(n, inplace=True)
   g['x'] = th.rand(n, 3)
   print(g['x'])
   
   # 1^{st} Approach: Basic gcn with no message passing
   # ## It can also be processed as if it is not a graph since edge information is not used
   gcn = GCN(nn.Linear(3, 2)) 
   g['x'] = gcn(g, g['x'])
   print(g['x'])
   
   # 2^{nd} Approach: Basic gcn with message passing via neighbors data summation
   g[(2, 3, 4)] = {2, 3, 4} ## Add edges to nodes 2, 3, 4; nodes can be removed via g.remove_edges()
   gcn = GCN(nn.Linear(2, 3), aggregation = sum_tensors)
   g['x'] = gcn(g, g['x'])
   print(g['x'])
   
   # 3^{rd} Approach: Proposed GCN with discovery and inclusion layers
   gcn = GCN(nn.Linear(3, 2), aggregation = sum_tensors, inclusion = nn.Linear(2, 3))
   g['x'] = gcn(g, g['x'])
   print(g['x'])
   ```

- Apply function

   ```python
   import lign as lg
   import torch as th
   from torch import nn
   from lign.nn import GCN
   n = 5
   g = lg.Graph().add(n, inplace=True)
   g['x'] = th.rand(n, 3)
   
   # Use apply if involving individual nodes
   ## Adds 3 to all node's 'x' data in the data set; doesn't require neighbors
   g.apply(lambda x: x + 3, data='x')
   print(g['x'])
   
   # Use push or pull if only involving multiple nodes. Nodes will push/pull data via edges
   ## Sums neighbors 'x' value together; require neighbors
   def sum_tensors(neighs):
       return th.stack(neighs).sum(dim = 0)
    
   g[(2, 3, 4)] = {2, 3, 4}
   g.push(sum_tensors, data='x')
   print(g['x'])
   ```

- Use clustering techniques for PyTorch

   ```python
   from lign.utils.clustering import NN, KNN, KMeans
   import torch as th
   
   n = 20
   x = th.rand(n, 3)
   labels = (x[:, 0] > 0.5)*1
   predict = th.rand(4, 3)
   print(predict)

   cluster = NN(x, labels)
   print(cluster(predict))

   cluster = KNN(x, labels, k=3)
   print(cluster(predict))

   cluster = KMeans(x, k=2)
   print(cluster(predict))
   ```

- Create sub graphs

   ```python
   import lign as lg
   import torch as th
   n = 5
   g = lg.Graph().add(n, inplace=True)
   g['x'] = th.rand(n, 3)
   g[tuple(range(n))] = set(range(3, n)) ## Add edge from each node to 3 and 4
   print(g)
   
   # Make sub graph with all data and edges from parent; edges are updated to reflect new indexes
   sub = g.sub_graph([2, 3, 4], get_data = True, get_edges = True)
   print(sub)
   
   # Make sub graph with only edges from parent; edges are updated to reflect new indexes
   sub = g.sub_graph(2, get_edges = True)
   sub.add(2)
   print(sub)
   
   # Make sub graph with only data from parent
   ## Add nodes not known to the parent graph
   sub = g.sub_graph([3, 4], get_data = True)
   sub.add([lg.node({'x': th.rand(3)}) for i in range(2)], self_loop = False)
   print(sub)
   
   sub[(2, 3)] = sub.get_parent_edges([0, 2])
   print(sub)
   ```

- Save and load created graphs

   ```python
   import lign as lg
   import torch as th
   n = 5
   g = lg.Graph().add(n, inplace=True)
   g['x'] = th.rand(n, 3)

   # Save to file
   g.save("data/graph.lign")
   
   # Load from file
   f = lg.Graph("data/graph.lign")

   # Check all data are the same
   print((f['x'] == g['x']).all())
   ```

- Convert common data set to lign graphs

   ```python
   import lign.utils as utl
   
   g0, g0_train, g0_validate = utl.load.mnist_to_lign("datasets/CIFAR100")
   
   g1, g1_train, g1_validate = utl.load.cifar_to_lign("datasets/CIFAR100")
   
   g2, g2_train, g2_validate = utl.load.cora_to_lign("datasets/CIFAR100")
   ```

_Please refer to the [documentation][examples-url] for other examples_

## Future

See the [open issues][issues-url] for a list of proposed features (and known issues).

## Citation

Refer to [CITATION.bib][citation-url] for BibTex citation

## Contributing

Read [CONTRIBUTING.md][contributing-url] for details on how to add to this repository.

_**tl;dr** Fork, create a new branch, commits features and make a pull request with documented changes_

## License

Distributed under the Mozilla Public License Version 2.0. See [LICENSE][license-url] for more information.

## Contact

[@josuecom_](https://github.com/JosueCom)

[contributors-shield]: https://img.shields.io/github/contributors/JosueCom/Lign.svg?style=for-the-badge
[contributors-url]: https://github.com/JosueCom/Lign/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JosueCom/Lign.svg?style=for-the-badge
[forks-url]: https://github.com/JosueCom/Lign/network/members
[stars-shield]: https://img.shields.io/github/stars/JosueCom/Lign.svg?style=for-the-badge
[stars-url]: https://github.com/JosueCom/Lign/stargazers
[issues-shield]: https://img.shields.io/github/issues/JosueCom/Lign.svg?style=for-the-badge
[issues-url]: https://github.com/JosueCom/Lign/issues
[license-shield]: https://img.shields.io/github/license/JosueCom/Lign.svg?style=for-the-badge
[license-url]: https://github.com/JosueCom/Lign/blob/master/LICENSE
[product-screenshot]: images/screenshot.png
[docs-url]: https://github.com/JosueCom/Lign/tree/master/docs
[examples-url]: https://github.com/JosueCom/Lign/tree/master/docs/examples
[bugs-url]: https://github.com/JosueCom/Lign/issues
[release-url]: https://github.com/JosueCom/Lign/releases
[issues-url]: https://github.com/JosueCom/Lign/issues
[repo-url]: https://github.com/JosueCom/Lign
[dev-docs-url]: https://github.com/JosueCom/Lign/tree/master/docs/dev
[citation-url]: https://github.com/JosueCom/Lign/blob/master/docs/CITATION.bib
[contributing-url]: https://github.com/JosueCom/Lign/blob/master/docs/CONTRIBUTING.md
[logo-url]: https://raw.githubusercontent.com/JosueCom/Lign/master/docs/imgs/logo.png
