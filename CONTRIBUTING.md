# Contributing to InstaGeo

First off, thank you for considering contributing to InstaGeo. It's people like you that make InstaGeo such a great tool.

### What should I know before I get started?

- Code of Conduct: InstaGeo adheres to a [Code of Conduct](#code-of-conduct). By participating, you are expected to uphold this code.
- Project Structure: Familiarize yourself with the layout of the project's repository. This can help you understand how the project is organized and where to make your changes.

### Ways to Contribute

There are many ways to contribute to InstaGeo, from writing tutorials or blog posts, improving the documentation, submitting bug reports and feature requests, or writing code which can be incorporated into InstaGeo itself.

#### Reporting Bugs

- Use the issue tracker to report bugs.
- Describe the bug and include additional details to help maintainers reproduce the problem.
- Provide a detailed description of the expected behavior.
- Include any relevant screenshots or error messages.

#### Suggesting Enhancements

- Use the issue tracker to suggest feature enhancements.
- Clearly describe the suggestion and include additional documentation or screenshots as necessary.

**!!** Before opening a new issue, make sure to search for keywords in the issues filtered by the
`"type::<TYPE>"` label and verify the issue you're about to submit isn't a duplicate.

#### Pull Requests

- Fork the repository and create your branch from `main`.
- If you've added code that should be tested, add tests.
- Ensure your code adheres to the existing style of the project to increase the chance of your changes being merged directly.
- Write a convincing description of your PR and why we should land it.


### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone.

## Contributing Code

All submissions, including submissions by project members, require review. We use GitHub pull
requests for this purpose. Consult
[GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
for more information on using pull requests.

Before sending your pull request for review, make sure your changes are consistent with the
guidelines and follow the coding style and code of conduct.

### General guidelines and philosophy for contribution

- Include unit tests when you contribute new features, as they help to a) prove that your code works
  correctly, and b) guard against future breaking changes to lower the maintenance cost.
- When you contribute a new feature to InstaGeo, the maintenance burden is (by default) transferred
  to the InstaGeo team. This means that the benefit of the contribution must be compared against the
  cost of maintaining the feature.
- Keep API compatibility in mind when you change code. Non-backward-compatible API changes will not
  be made if they don't greatly improve the library.
- As every PR requires CI testing, we discourage submitting PRs to fix one typo, one warning, etc.
  We recommend fixing the same issue at the file level at least (e.g.: fix all typos in a file, fix
  all compiler warnings in a file, etc.)

### Coding Style

To guarantee the quality and uniformisation of the code, we use various linters:

- [Black](https://black.readthedocs.io/en/stable/#) is a deterministic code formatter that is
  compliant with PEP8 standards.
- [Isort](https://pycqa.github.io/isort/) sorts imports alphabetically and separates them into
  sections.
- [Flake8](https://flake8.pycqa.org/en/latest/) is a library that wraps PyFlakes and PyCodeStyle. It
  is a great toolkit for checking your codebase against coding style (PEP8), programming, and syntax
  errors. Flake8 also benefits from an ecosystem of plugins developed by the community that extend
  its capabilities. You can read more about Flake8 plugins on the documentation and find a curated
  list of plugins here.
- [MyPy](https://mypy.readthedocs.io/en/stable/#) is a static type checker that can help you detect
  inconsistent typing of variables.

### Pre-Commit

To help in automating the quality of the code, we use [pre-commit](https://pre-commit.com/), a
framework that manages the installation and execution of git hooks that will be run before every
commit. These hooks help to automatically point out issues in code such as formatting mistakes,
unused variables, trailing whitespace, debug statements, etc. By pointing these issues out before
code review, it allows a code reviewer to focus on the architecture of a change while not wasting
time with trivial style nitpicks. Each commit should be preceded by a call to pre-commit to ensure
code quality and formatting. The configuration is in .pre-commit-config.yaml and includes Black,
Flake8, MyPy and checks for the yaml formatting, trimming trailing whitespace, etc. Try running:
`pre-commit run --all-files`. All linters must pass before committing your change.

### Docstrings

Public modules, functions, classes, and methods must be documented using Python docstrings. These
docstrings must have sections for Arguments, Returns, and Raises (if applicable). For every argument
of a function, the docstring must explain precisely what the argument does, what data type it
expects, whether or not it is optional, and any requirements for the range of values it expects. The
same goes for the returns. Use existing docstrings as templates. Non-public functions and methods
must also be documented for defining the API contract. In addition to being useful for generating
documentation, docstrings add clarity when looking through the source code or using the built-in
help system, and can be leveraged in autocompletion by IDEs.

Please see [PEP 257](https://peps.python.org/pep-0257/) for details on semantics and conventions
associated with Python docstrings. Docstrings must follow
[Google style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
This docstring style is more pleasant to read when browsing the source.

If you are using Visual Studio Code, you can install the
[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)
extension to quickly generate docstring snippets.

### Code of Conduct

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

### Testing

Please make sure that your PR passes all tests by running
[pytest](https://docs.pytest.org/en/latest/) on your local machine. Also, you can run only tests
that are affected by your code changes, but you will need to select them manually.

### Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement. You (or your
employer) retain the copyright to your contribution, this simply gives us permission to use and
redistribute your contributions as part of the project. Head over to
https://cla-assistant.io/instadeepai/ to see your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one (even if it was for
a different project), you probably don't need to do it again.
