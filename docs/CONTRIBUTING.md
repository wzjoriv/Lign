# Contributing

## Table of Contents

- [Contributing](#contributing)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Making Changes](#making-changes)
  - [Documentation](#documentation)
    - [Pull Request](#pull-request)
    - [Issue](#issue)
    - [Feature](#feature)
    - [Docstring](#docstring)
  - [Developer Docs](#developer-docs)

## Getting Started

These are instructions for getting started with development. If only interested in the package, please install LIGN via ``pip install lign`` or by downloading the [package][release-url] on github.

_For more details, please read the [development documentation](docs/dev)_

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

   ```sh
   pip install -r docs/dev/requirements.txt
   ```

4. Install package

   ```sh
   pip install .
   ```

### Making Changes

When contributing to the LIGn, please make sure to follow these instruction as a documenting any chnages made.

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

If you create a pull request, you must document any changes made. There are four types of documentations that may be made based on the extend of the contribution.

### Pull Request

When contributing any change, please insert and fill the following template in the pull request description.

```text
Changes: 
    - [file1.rm] - [made change 1 and 2]
    - [sub/folder/file2.py] - [made another change here]

Reason:
    - [I made change 1 and 2 to make an example]
    - [Made another change to show how to reference nested files]

Description (Optional):
    [Use description to describe long changes that are longer than a sentence, involve multiple files or provide greater detail]

Comment (Optional):
    [Use this template for pull requests]
```

### Issue

When reporting an issue, please insert and fill the following template in the description. Additionally, insert ``[ISSUE]`` or ``[BUG]`` in the title of the Github Issue.

_Create one issue for each independent problem._
_Do not group multiple unrelated issues into one report_

```text
Issue: 
    - [One or two sentences about the issue]

Replicate:
    - [what to do to]
    - [What else to do]
    ```
        Feel free to add code to replicate issue
    ```

Description (Optional):
    [Use description to describe long issue that are longer than a sentence, involve multiple files or provide greater detail]

Output (Optional):
    ```
        Feel free to add console output
    ```
    or 
    [Screenshots](image.png)

Comment (Optional):
    [Use this template for issues]
```

### Feature

When requesting a feature, please insert and fill the following template in the description. Additionally, insert ``[FEATURE]`` or ``[IMPROVEMENT]`` in the title of the Github Issue.

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

Feel free to read the [developer documentation](docs/dev) for more information about contributing to LIGN.

[release-url]: https://github.com/JosueCom/LIGN/releases
[example-url]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google
[docstring-url]: https://google.github.io/styleguide/pyguide.html
