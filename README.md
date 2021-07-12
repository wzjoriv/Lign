# LIGN

[![LIGN logo](docs/imgs/logo.png "LIGN logo")][repo-url]

A deep learning framework for implementing lifelong learning and graph convoluitional networks (GCN) on tensor and graph data

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

- [LIGN](#lign)
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

LIGN is a deep learning framework developed for implementing continual lifelong learning, graph convolutional networks (GCN) and graph-based machine learning. It was build as a component of my master thesis.

### Built With

- [PyTorch](https://pytorch.org)

## Getting Started

These are instructions for getting started with development. If only interested in the package, please install LIGN via ``pip install lign`` or by downloading the [package][release-url] on github.

_For more details, please read the [developer documentation](docs/dev)_

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/JosueCom/LIGN.git
   ```

2. Switch directory

   ```sh
   cd LIGN
   ```

3. Install prerequisites

   _It is recommended to install [PyTorch](https://pytorch.org) via the official site first for optimal performance_

   ```sh
   pip install -r docs/dev/requirements.txt
   ```

   ```sh
   conda env create -f docs/dev/environment.yml -n LIGN
   conda activate LIGN
   ```

4. Install package

   ```sh
   pip install . --upgrade
   ```

## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_Please refer to the [documentation](docs/examples) for other examples_

## Future

See the [open issues][issues-url] for a list of proposed features (and known issues).

## Citation

Refer to [CITATION](docs/CITATION.bib) for BibTex citation

## Contributing

Read [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on how to add to this repository.

_**tl;dr** Fork, create a new branch, commits features and make a pull request with documented changes_

## License

Distributed under the Mozilla Public License Version 2.0. See [LICENSE](LICENSE) for more information.

## Contact

[@josuecom_](https://github.com/JosueCom)

[contributors-shield]: https://img.shields.io/github/contributors/JosueCom/LIGN.svg?style=for-the-badge
[contributors-url]: https://github.com/JosueCom/LIGN/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/JosueCom/LIGN.svg?style=for-the-badge
[forks-url]: https://github.com/JosueCom/LIGN/network/members
[stars-shield]: https://img.shields.io/github/stars/JosueCom/LIGN.svg?style=for-the-badge
[stars-url]: https://github.com/JosueCom/LIGN/stargazers
[issues-shield]: https://img.shields.io/github/issues/JosueCom/LIGN.svg?style=for-the-badge
[issues-url]: https://github.com/JosueCom/LIGN/issues
[license-shield]: https://img.shields.io/github/license/JosueCom/LIGN.svg?style=for-the-badge
[license-url]: https://github.com/JosueCom/LIGN/blob/master/LICENSE
[product-screenshot]: images/screenshot.png
[docs-url]: https://github.com/JosueCom/LIGN/tree/master/docs
[examples-url]: https://github.com/JosueCom/LIGN/tree/master/docs/examples
[bugs-url]: https://github.com/JosueCom/LIGN/issues
[release-url]: https://github.com/JosueCom/LIGN/releases
[issues-url]: https://github.com/JosueCom/LIGN/issues
[repo-url]: https://github.com/JosueCom/LIGN
