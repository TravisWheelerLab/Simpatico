# Project

**REPLACE:** One sentence description of the project.

## About

**REPLACE:** Describe your project in more detail. What does it do? Who are the intended
users? Why is it important or meaningful? How does it improve upon similar
software? Is it a component of or extension to a larger piece of software?

## Installation

git clone git@github.com:TravisWheelerLab/Simpatico.git
pip install Simpatico

## Usage
Below, we demonstrate the training and evaluation process, using pdbbind for training and the DUDE decoy screening for evaluation.

### Wrangling Training Data

#### Converting molecular structures to PyG graphs
Simpatico is trained on structural data of resolved protein-ligand complexes as described in .pdb files. Therefore, our first step is to process the .pdb files we'd like to use as training data to generate protein-pocket and small-molecule graphs and conveniently store them as PyG graph objects. 

As with many of simpatico's functions, we may pass in a list of paths to the structure files, or a quote-enclosed unix path, "/something/like/this.pdb". The files pointed to with either method will be converted to PyG graph objects and stored in the specified output location. 

The crucial difference between converting protein and molecule files is the use of the `-p` or `-m` to indicate the type of structure.

```bash
   % simpatico convert -i "$HOME/data/pdbbind/*/*_protein.pdb" -p -o train_target_pygs/

   % simpatico convert -i "$HOME/data/pdbbind/*/*_ligand.sdf" -m -o train_ligand_pygs/
```

#### Train-validation sets
Now we need to organize these graphs into a train-validation set to pass on to our training process. This requires a `.txt` file structured as follows:

```
t, path/to/protein/target/graph.pyg, path/to/corresponding/ligand/graph.pyg
t, path/to/protein/target/graph.pyg, path/to/corresponding/ligand/graph.pyg
t, path/to/protein/target/graph.pyg, path/to/corresponding/ligand/graph.pyg
v, path/to/protein/target/graph.pyg, path/to/corresponding/ligand/graph.pyg
v, path/to/protein/target/graph.pyg, path/to/corresponding/ligand/graph.pyg
v, path/to/protein/target/graph.pyg, path/to/corresponding/ligand/graph.pyg
```

All target-ligand pairs below the `training` but above the `validation` line will be trained on to optimize the weights. all target-ligand pairs described 



### Training




### Evaluation




## Development

**REPLACE:** What language(s) are in use? What does a user need to install for development
purposes? This might include build systems, Docker, a particular compiler or
runtime version, test libraries or tools, linters, code formatters, or other
tools. Are there any special requirements, like processor architecture? What
commands should developers use to accomplish common tasks like building, running
the test suite, and so on?

## Contributing

We welcome external contributions, although it is a good idea to contact the
maintainers before embarking on any significant development work to make sure
the proposed changes are a good fit.

Contributors agree to license their code under the license in use by this
project (see `LICENSE`).

To contribute:

  1. Fork the repo
  2. Make changes on a branch
  3. Create a pull request

## License

See `LICENSE` for details.

## Authors

**REPLACE:** Who should people contact with questions?

See `AUTHORS` the full list of authors.

### Branch
Preprint Refactor
