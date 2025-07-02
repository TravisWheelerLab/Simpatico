# Project

**REPLACE:** One sentence description of the project.

## About

**REPLACE:** Describe your project in more detail. What does it do? Who are the intended
users? Why is it important or meaningful? How does it improve upon similar
software? Is it a component of or extension to a larger piece of software?

## Installation
<details closed>
<summary><strong>Installing simpatico on your system</strong></summary>
Simpatico depends on several GPU-based libraries such as PyTorch that are sensitive to the variables of different computing environments, things like CUDA version or GPU availability. These dependencies include PyTorch, PyG, and Faiss.

The following sequence of commands is likely to accommodate most users. This procedure has been cobbled together from installation instructions provided by the respective libraries at the following URLs:
  - PyTorch - https://pytorch.org/get-started/locally/
  - PyG - https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
  - Faiss - https://pypi.org/project/faiss-gpu-cu12/

### 1. Installing PyTorch
```bash 
$ pip install torch torchvision torchaudio
```
Now to identify PyTorch and CUDA versions:

```bash
$ python -c "import torch; print(torch.__version__)"
```

Which will produce a value like:
```bash 
$ 2.7.0+cu126
```
If this fails to produce a `cu{NUMBER}` value, try 
```bash
$ python -c "import torch; print(torch.version.cuda)"
```

### 2. Installing PyG
Now install the correct PyG packages according to the provided version numbers:
```bash
$ pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```
Note that values of ``torch-2.7.0+cu126.html`` need to be changed to match the version numbers produced by the previous command.

### 3. Installing Faiss
Finally, we can install the Faiss-gpu library that matches our version of CUDA:

```bash
$ pip install faiss-gpu-cu12
```

### 4. Installing simpatico

With all the dependencies in place, installing simpatico itself is quite simple!

```bash
$ git clone git@github.com:TravisWheelerLab/Simpatico.git
$ pip install Simpatico
```
</details>

## Usage
<details closed>
<summary><strong>Generating protein and small molecule atom embeddings</strong></summary>

Out of the box, simpatico comes with weights trained on PDBBind, and may be used to generate embeddings for your very own proteins and small molecules. 

To obtain embeddings for a protein pocket, a .csv of the following format must be specified:

**protein_eval_input.csv**:
```
/path/to/protein_1.pdb, /path/to/pocket_coordinates_1.sdf
/path/to/protein_2.pdb, /path/to/pocket_coordinates_2.sdf
...
```
In this example, the positions of the ligand items will be used to specify the protein pocket surface atoms. Pockets may be specified with any 3D molecular structure file (.sdf, .mol2, or .pdb), or with a 3-column .csv file where each line represents a new X,Y,Z coordinate.

Having a method for pointing to our target structure files and specifying pockets, we may generate embeddings via the following:

### Command Usage 
```bash
$ simpatico eval INPUT_FILE.csv OUTPUT_DIR/ (-p | -m)
```
Where either `-p` or `-m` must be specified to indicate `protein` or `molecule`.
### Example 
```bash
$ simpatico eval protein_eval_input.csv protein_embeds/ -p
```
For each protein .pdb file supplied, a .pyg file will be produced in the specified output directory. The nodes of this graph represent pocket surface atoms, with embedding values stored in `graph.x`, and 3D positions in `graph.pos`.

Generating small-molecule embeds is a nearly identical process, except the input .csv file requires only one column:
**molecule_eval_input.csv**:
```
/path/to/molecule_1.sdf
/path/to/molecule_2.sdf
...
```
So running the command will look like:
### Example 
```bash
$ simpatico eval molecule_eval_input.csv molecule_embeds/ -m
```
For each specified molecular structure, a .pyg file representing a [batch of graphs](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.data.Batch.html) will be generated. The batch contains all molecules described in the sdf file. Like the protein embedding output, the embedding values are contained in `graph.x`.
</details>
<details closed>
<summary><strong>Querying the vector database</strong></summary>
In the [simpatico paper](https://www.biorxiv.org/content/10.1101/2025.06.08.658499v1), we perform virtual screening by using protein pocket embeddings to query a Faiss vector database of small molecular embeddings, then performing a simple aggreagation procedure over the queries' nearest neighbors. 

To perform this operation, you'll need a .csv which embedding (.pyg) files will serve as  
</details>

## Training
<details closed>
<summary><strong>Generating train-test splits</strong></summary>
</details closed>

<details closed>
<summary><strong>Training and updating model weights</strong></summary>
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

</details>

## Evaluation
<details closed>
<summary><strong>Testing on a decoy database</strong></summary>
</details>

Below, we demonstrate the training and evaluation process, using pdbbind for training and the DUDE decoy screening for evaluation.




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
