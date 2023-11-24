# 🧩 Machine Learning Research Project Template

## 👀 Overview

With the growth of our research activities, our team started experiencing inconsistencies in our coding practices and tools.
Researchers needed to copy-paste boilerplate codes between repositories, and couldn't easily benefit from the utils created in other projects.
This was the source of wasted time and energy on low value-add works and hindered collaborations between projects.
Our objective with this template is therefore to help our researchers to:

1. Reduce the time to set up a new repository and the time-to-experimentation.
2. Share the collective knowledge acquired while building up our software activities.

In practice, this template:

1. Provides a generic structure to organise your code.
2. Provide boilerplate codes, i.e. pre-commit and linters, CI, license, documentation, standard dockerfile and makefile, gitignore, AIchor manifest, etc.
3. Put forward a set of good practices and guidelines that have demonstrated their benefits.

In this quest to facilitate your setup, assumptions have been made.
For instance, we assume here that your work will be done in Python (3.8), that you will use JAX as your Deep Learning library, and will run your code either on AIchor and/or GCP VM (GPU, TPU).
Be aware that these assumptions might not hold for your project, and adjustments should be made.

### 🎃 Out of Scope

It is tempting to add as much as possible to this repository. We may indeed be slightly too generous in our offering,
especially when the feature can be easily ignored or removed (e.g. CI pipeline).
However, we should keep a strong focus on our objective and not deviate by trying to incorporate too many elements in this single template.
In particular, we do not seek to:
- Push for production-level code, i.e. testing, release process, strict type hinting, etc.
- Provide all the components required to open-source the code.

### 🤝 Contribution

Contributions are welcome!
This template must be an up-to-date reflection of our practices and preferred tooling.
If you spot a problem or think of an improvement, don’t wait. Draft a PR and share it with your colleagues.

## 🚀 Getting started

This template already includes certain example workflows that will often be repeated between many, or all, projects.
In particular the [Makefile](Makefile) is configured for ease of building [Docker](docker) images,
starting containers and launching experiments locally and on both single- and multi-node TPU devices.

**Local quickstart**
```commandline
make docker_build_cpu
make docker_run command="python experiments/run.py"
```

**TPU VM**

Edit the [Makefile](Makefile) with the desired TPU configuration.

```commandline
export GITHUB_USER_TOKEN={TOKEN} GITHUB_ACCESS_TOKEN={TOKEN}
make create
make run command="cd ml-research-template; make docker_build_tpu"
make run command="cd ml-research-template; make docker_run command='python experiments/run.py'"
```
