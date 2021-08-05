# Contributing

Please follow the instructions below for contributing to lign. Contributions that do not follow the guidelines below may be removed or closed at the discretion of the reviewer.

## Table of Contents

- [Contributing](#contributing)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Making Changes](#making-changes)
  - [Documentation](#documentation)
    - [Pull Request](#pull-request)
    - [Issue](#issue)
    - [Feature Request](#feature-request)
    - [Docstring](#docstring)
  - [Developer Docs](#developer-docs)

## Getting Started

These are instructions for getting started with development. If only interested in the package, please install Lign via ``pip install lign``, ``conda install lign -c josuecom -c pytorch`` or by downloading the [package][release-url] from github and saving it in the ``site-packages/`` directory of your python interpreter.

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

### Making Changes

When contributing to the Lign, please make sure to follow these instruction and document any changes made.

1. Fork the project
2. Create your feature branch

   ```sh
   git checkout -b feature/amazing-feature
   ```

3. Commit your Changes

   ```sh
   git commit -m 'Add some AmazingFeature'
   ```

4. Push to the Branch

   ```sh
   git push origin feature/AmazingFeature
   ```

5. Open a Pull Request

## Documentation

If you contribute to lign, you must follow the following standards and document any changes made as described below. There are four types of documentations that may be made based on the extend of the contribution.

### Pull Request

When contributing any changes, first document your code by following the convention under [Docstring](#docstring). After, create a pull request. Lastly, insert and fill the following template in the pull request description.

```text
Changes: 
    - [made change 1 and 2]
    - [made another change here]

Reason:
    - [I made change 1 and 2 to make an example]
    - [Made another change to show how to reference nested files]

Description (Optional):
    [Use description to describe long changes that are longer than a sentence, involve multiple files or provide greater detail]

Comment (Optional):
    [Use this template for pull requests]
```

### Issue

When reporting an issue, use the discussion panel on GitHub to discuss it with the community first. If (1)  no response is given after 24 hours or an inarticulate respond was given, (2) the issue is considered novel and (3) no duplicate issue(s) exist, feel free to submit a request by creating an issue with the following template. Additionally, insert ``[ISSUE]`` or ``[BUG]`` in the title of the GitHub Issue. Contributions that do not follow these guidelines may be removed or closed at the discretion of the reviewer.

Create one issue for each independent problem. Do not group multiple unrelated issues into one report.

[Template][issue-template]

### Feature Request

When requesting a feature, use the discussion panel on Github to discuss it first with the community. __After 24 hours__, some __considerations__, if the feature is __considered beneficial__ and __no duplicate__ functionality or request exists, please submit the request by creating an issue then inserting and filling the following template in the description. Additionally, insert ``[FEATURE]`` or ``[IMPROVEMENT]`` in the title of the Github Issue.

_Create one feature for each independent improvement._
_Do not group multiple unrelated features into one report_

```text
Feature:
    - [One or two sentences about the feature]

Reason:
    - [This feature will make bla faster]

Description (Optional):
    [Use description to describe long features that are longer than a sentence, involve multiple files or provide greater detail]

Comment (Optional):
    [Use this template for features]
```

### Docstring

Any changes made to the source code (``ling/``) must be documented using [Google style python docstrings][docstring-url]. Refer to [napoleon readthedocs][example-url] for an example of Google style python docstrings.

## Developer Docs

Feel free to read the [developer documentation](docs/dev) for more information about contributing to Lign.

[release-url]: https://github.com/JosueCom/Lign/releases
[example-url]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google
[docstring-url]: https://google.github.io/styleguide/pyguide.html
[issue-template]: https://github.com/JosueCom/Lign/blob/master/.github/ISSUE_TEMPLATE/issue-report.md
