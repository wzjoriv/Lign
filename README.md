# Lign

[![Lign logo](docs/imgs/logo.png "Lign logo")][repo-url]

A deep learning framework developed for implementing graph convolutional networks (GCNs), continual lifelong learning on graphs and other graph-based machine learning methods alongside PyTorch

[View Docs][docs-url]
·
[View Examples][examples-url]
·
[Report Bugs][bugs-url]
·
[Request Feature][bugs-url]

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Mozilla License][license-shield]][license-url]

## Table of Contents

- [Lign](#lign)
  - [Table of Contents](#table-of-contents)
  - [About The Project](#about-the-project)
    - [Built With](#built-with)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Future](#future)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

## About The Project

Lign is a deep learning framework developed for implementing graph convolutional networks (GCNs), continual lifelong learning on graphs and other graph-based machine learning methods alongside PyTorch. It was build as a component of my master thesis.

### Built With

- [PyTorch](https://pytorch.org)

## Getting Started

These are instructions for getting started with development. If only interested in the package, please install Lign via ``pip install lign``, ``conda install lign -c pytorch -c josuecom`` or by downloading the [package][release-url] on github.

_For more details, please read the [developer documentation](docs/dev)_

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

* Create a new graph, assign some values and view properties

   ```python
   >>> import lign as lg
   >>> import torch as th
   >>> n = 5
   >>> g = lg.Graph()
   >>> g.add(n)
   >>> g['x'] = th.rand(n, 3) ## e.g. g.set_data('x', th.rand(n, 3))
   >>> 
   >>> g
   >>> g['x'] ## e.g. g.get_data('x')
   >>> g[0] ## e.g. g.get_nodes(1)
   >>> g[[1, 2]] ## e.g. g.get_nodes([1, 2])
   >>> g[3:4] ## e.g. g.get_nodes(slice(3, 4))
   >>> g[(4,)] ## e.g. g.get_edges(4)
   ```

* Process data with a neural network

   ```python
   >>> import lign as lg
   >>> import torch as th
   >>> from torch import nn
   >>> n = 5
   >>> g = lg.Graph()
   >>> g.add(n, add_edges=False) ## not self loop edges since no relational data present
   >>> g['x'] = th.rand(n, 3) ## e.g. g.set_data('x', th.rand(n, 3))
   >>> g['x']
   >>> 
   >>> linear = nn.Linear(3, 2)
   >>> g['x'] = linear(g['x']) ## e.g. g.apply(linear, data = 'x')
   >>> g['x']
   ```

* Process data with a GCN

   ```python
   >>> import lign as lg
   >>> import torch as th
   >>> from torch import nn
   >>> from lign.layers import GCN
   >>> from lign.utils.functions import sum_neighs_data
   >>> n = 5
   >>> g = lg.Graph()
   >>> g.add(n)
   >>> g['x'] = th.rand(n, 3) ## e.g. g.set_data('x', th.rand(n, 3)
   >>> g['x']
   >>> 
   >>> gcn = GCN(nn.Linear(3, 2))
   >>> g['x'] = gcn(g, g['x'])
   >>> g['x']
   >>> 
   >>> g[(2, 3, 4)] = {2, 3, 4} ## Add edges to nodes 2, 3, 4; remove via g.remove_edges()
   >>> gcn = GCN(nn.Linear(2, 3), aggregation = sum_neighs_data)
   >>> g['x'] = gcn(g, g['x'])
   >>> g['x']
   >>> 
   >>> gcn = GCN(nn.Linear(3, 2), aggregation = sum_neighs_data, inclusion = nn.Linear(2, 3))
   >>> g['x'] = gcn(g, g['x'])
   >>> g['x']
   ```

* Apply function

   ```python
   >>> import lign as lg
   >>> import torch as th
   >>> from torch import nn
   >>> from lign.layers import GCN
   >>> n = 5
   >>> g = lg.Graph()
   >>> g.add(n)
   >>> g['x'] = th.rand(n, 3) ## e.g. g.set_data('x', th.rand(n, 3))
   >>> 
   >>> add_constant = lambda x: x + 3
   >>> g.apply(add_constant, data='x') ## use apply if involving individual nodes
   >>> g['x']
   >>> 
   >>> def sum_neighs_data(neighs):
   ...     out = neighs[0]
   ...     for neigh in neighs[1:]:
   ...         out = out + neigh
   ...     return out
   ... 
   >>> g[(2, 3, 4)] = {2, 3, 4} ## Add edges to nodes 2, 3, 4; remove via g.remove_edges()
   >>> g.push(sum_neighs_data, data='x') ## use push or pull if only involving multiple nodes
   >>> g['x']
   ```

* Create subgraphs

   ```python
   >>> import lign as lg
   >>> import torch as th
   >>> n = 5
   >>> g = lg.Graph()
   >>> g.add(n)
   >>> g['x'] = th.rand(n, 3) ## Others, g.set_data('x', th.rand(n, 3))
   >>> g[tuple(range(n))] = set(range(3, n))
   >>> g
   >>> 
   >>> sub = g.subgraph([2, 3, 4], get_data = True, get_edges = True)
   >>> sub
   >>> 
   >>> sub = g.subgraph(2, get_edges = True)
   >>> sub.add(2)
   >>> sub
   >>> 
   >>> sub = g.subgraph([3, 4], get_data = True)
   >>> sub.add([lg.node({'x': th.rand(3)}) for i in range(2)], add_edges = False)
   >>> sub
   >>> sub[(2, 3)] = sub.get_parent_edges([0, 2])
   >>> sub
   ```

* Save and load created graphs

   ```python
   >>> import lign as lg
   >>> import torch as th
   >>> n = 5
   >>> g = lg.Graph()
   >>> g.add(n)
   >>> 
   >>> g['x'] = th.rand(n, 3) ## Others, g.set_data('x', th.rand(n, 3))
   >>> g.save("data/graph.lign")
   >>> 
   >>> f = lg.Graph("data/graph.lign") ## load from file
   >>> (f['x'] == g['x']).all() ## are all data the same check
   ```

_Please refer to the [documentation](docs/examples) for other examples_

## Future

See the [open issues][issues-url] for a list of proposed features (and known issues).

## Citation

Refer to [CITATION.bib](docs/CITATION.bib) for BibTex citation

## Contributing

Read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on how to add to this repository.

_**tl;dr** Fork, create a new branch, commits features and make a pull request with documented changes_

## License

Distributed under the Mozilla Public License Version 2.0. See [LICENSE](LICENSE) for more information.

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
