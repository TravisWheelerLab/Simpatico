# Project
Simpatico is a graph neural network for producing high-dimensional embeddings of atoms in proteins and small molecules. Atomic representations produced by Simpatico are co-located in embedding space according to their interaction potential. This allows users to perform rapid virtual screening over extremely large datasets. In [our paper](https://www.biorxiv.org/content/10.1101/2025.06.08.658499v1), we show that not only is Simpatico’s binding prediction accuracy competitive with state-of-the-art deep learning-assisted docking methods, but it can perform virtual screening more than 1000× faster. Furthermore, Simpatico embeddings are versatile: users may just as easily use them to screen protein pockets with a small-molecule target (akin to toxicology screening), or to assess shared binding properties between non-homologous protein structures.

## Installation

<details closed>
<summary><strong>Installing simpatico on your system</strong></summary>
Simpatico depends on several GPU-based libraries such as PyTorch that are sensitive to your computing environment (e.g., CUDA version and GPU availability). These dependencies include PyTorch, PyG, and Faiss.


The following sequence of commands will work for most users. This procedure is assembled from installation instructions provided by the respective libraries:

* PyTorch – [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
* PyG – [https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
* Faiss – [https://pypi.org/project/faiss-gpu-cu12/](https://pypi.org/project/faiss-gpu-cu12/)

### 1. Installing PyTorch

```bash
pip install torch
```

Verify your PyTorch and CUDA versions:

```bash
python -c "import torch; print(torch.__version__)"
```

This will produce a value like:

```bash
2.7.0+cu126
```

If you don’t see a `cu{NUMBER}` value, try:

```bash
python -c "import torch; print(torch.version.cuda)"
```

### 2. Installing PyG

Install the correct PyG packages according to the version numbers from the previous command:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.0+cu126.html
```

**Important:** Replace `torch-2.7.0+cu126.html` with your specific PyTorch and CUDA version.

### 3. Installing Faiss

Install the Faiss GPU library matching your CUDA version:

```bash
pip install faiss-gpu-cu12
```

Again, adjust `cu12` to correspond to your CUDA version if needed.

### 4. Installing simpatico

With all dependencies installed, you can now install simpatico itself:

```bash
git clone git@github.com:TravisWheelerLab/Simpatico.git
pip install Simpatico
```

</details>

## Usage

For demonstration purposes, example protein and small molecule structures are included in `Simpatico/examples/data/`, including a small random selection from PDBBind and a tiny compound library sourced from ENAMINE. In the examples below, commands are run from the root `Simpatico` directory.

<details closed>
<summary><strong>Generating protein and small molecule atom embeddings</strong></summary>

Out of the box, simpatico comes with weights trained on PDBBind and can be used to generate embeddings for your own proteins and small molecules.

To obtain embeddings for protein pockets, prepare a CSV file in this format:

**Simpatico/examples/spec\_files/protein\_eval\_example.csv**

```
examples/data/pdbbind_sample/1a7c/1a7c_pocket.pdb, examples/data/pdbbind_sample/1a7c/1a7c_ligand.sdf
examples/data/pdbbind_sample/1a7x/1a7x_pocket.pdb, examples/data/pdbbind_sample/1a7x/1a7x_ligand.sdf
examples/data/pdbbind_sample/1ahx/1ahx_pocket.pdb, examples/data/pdbbind_sample/1ahx/1ahx_ligand.sdf
...
```

Positional data from the ligand files in the second column will be used to define the protein pocket surface atoms. Pockets can be specified with any 3D molecular structure file (`.sdf`, `.mol2`, `.pdb`) or with a 3-column CSV where each line lists an X, Y, Z coordinate.

Once your target structure files are ready, generate embeddings with:

### Command Usage

```bash
simpatico eval INPUT_FILE.csv OUTPUT_DIR/ (-p | -m)
```

You must specify either `-p` (protein) or `-m` (molecule).

### Example

```bash
simpatico eval examples/spec_files/protein_eval_example.csv examples/data/protein_embeds -p
```

For each protein `.pdb` file, a `.pyg` file is created in the output directory. The graph nodes represent pocket surface atoms, with embedding values stored in `graph.x` and 3D positions in `graph.pos`.

Generating small molecule embeddings is nearly identical. In this case, the input CSV requires only a single column:

**Simpatico/examples/spec\_files/mol\_eval\_example.csv**

```
examples/data/smiles_sample/smiles_1.ism
examples/data/smiles_sample/smiles_2.ism
examples/data/smiles_sample/smiles_3.ism
...
```

Run the command as follows:

### Example

```bash
simpatico eval examples/spec_files/mol_eval_example.csv examples/data/mol_embeds -m
```

For each specified molecule, a `.pyg` file containing a [batch of graphs](https://pytorch-geometric.readthedocs.io/en/2.5.3/generated/torch_geometric.data.Batch.html) will be generated. Embedding values are stored in `graph.x`, similar to protein embeddings.

</details>
<details closed>
<summary><strong>Querying the vector database (virtual screening)</strong></summary>
In the [simpatico paper](https://www.biorxiv.org/content/10.1101/2025.06.08.658499v1), we demonstrate virtual screening by using protein pocket embeddings to query a Faiss vector database of small molecule embeddings, followed by an aggregation procedure over each query’s nearest neighbors.

To run a query, prepare a CSV file specifying which embedding files to use as queries (e.g., protein pockets) and which as the vector database (e.g., candidate molecules). Example:

**examples/spec\_files/query\_example.csv**

```
q,examples/data/protein_embeds/2fme_pocket_embeds.pyg
q,examples/data/protein_embeds/5m4k_pocket_embeds.pyg
...
d,examples/data/mol_embeds/smiles_2_embeds.pyg
d,examples/data/mol_embeds/smiles_1_embeds.pyg
```

Each line has two columns: the first is `q` (query) or `d` (database), and the second is the path to a `.pyg` file created by `simpatico eval`.

Run the query:

```bash
simpatico query examples/spec_files/query_example.csv examples/data/query_results.pkl
```

This generates score values saved to `examples/data/query_results.pkl`.

To get a human-readable version of the results, run:

```bash
simpatico print-results examples/data/query_results.pkl
```

The output is structured as follows:

```
>target sources:
1 examples/data/pdbbind_sample/2fme/2fme_pocket.pdb
2 examples/data/pdbbind_sample/5m4k/5m4k_pocket.pdb
...

>db sources:
1 examples/data/smiles_sample/smiles_2.ism
2 examples/data/smiles_sample/smiles_1.ism
...

>results:
1 1 2 827 1
1 1 1 815 2
1 1 1 489 3
...
16 1 3 196 61
16 1 1 784 62
...
```

Each block lists indices corresponding to the query and database files.

The lines below `>results:` list the top scoring matches, using this column format:

```
TARGET_SOURCE_INDEX TARGET_SOURCE_ITEM DB_SOURCE_INDEX DB_SOURCE_ITEM ITEM_RANK
```

For example, the line:

```
1 1 2 827 1
```

This line is read as: the best-scoring molecule (`ITEM_RANK=1`) for item 1 from target file 1 (`2fme_pocket.pdb`) comes from database file 2 (`smiles_1.ism`), specifically the 827th molecule in that file.

Farther down, the line:

```
16 1 3 196 61
```

For query file 16, item 1, the 61st highest-scoring molecule is molecule 196 in database file 3.

Results may be saved to a `.txt` file or some other output by sending the output to the desired file, like:

```bash
simpatico print-results examples/data/query_results.pkl > vs_results.txt
```
</details>

## Training
<details closed>
<summary><strong>Training and updating model weights</strong></summary>
Simpatico is trained on structural data of protein-ligand complexes. Each training sample consists of one protein structure and one ligand structure, which together make up the bound protein-ligand complex. The first step in training or fine-tuning a model will be to specify which protein structure files correspond to which ligand structure files. We must also denote which of these pairs should be included in the training set, and which should be held out for the validation set. This is performed by listing the files accordingly in the `.csv` file we will ultimately be providing the training function. 

**examples/spec\_files/query\_example.csv**
```
t, examples/data/pdbbind_sample/1a7c/1a7c_pocket.pdb, examples/data/pdbbind_sample/1a7c/1a7c_ligand.sdf
t, examples/data/pdbbind_sample/1a7x/1a7x_pocket.pdb, examples/data/pdbbind_sample/1a7x/1a7x_ligand.sdf
...
v, examples/data/pdbbind_sample/6v1c/6v1c_pocket.pdb, examples/data/pdbbind_sample/6v1c/6v1c_ligand.sdf
v, examples/data/pdbbind_sample/8lpr/8lpr_pocket.pdb, examples/data/pdbbind_sample/8lpr/8lpr_ligand.sdf
```

The value in the first column will be either `t` or `v` to indicate whether the protein-ligand pair belongs to the train or validation set, respectively. The second column is the path to the protein pdb structure, and the third column the path to the corresponding ligand structure. 

With the proper `.csv` file, you may now kickoff a new round of training with the following command:

```bash
simpatico train examples/spec_files/train_example.csv examples/data/example_weights.pt -o examples/data/example_train.out -l simpatico/models/weights/model_v1.pt
```

In this example, we are loading the pretrained weights `-l simpatico/models/weights/model_v1.pt` and updating them with our new training examples. Note that this is a naive approach to fine-tuning, and we have not yet implemented regularization techniques appropriate for proper fine-tuning.  

</details closed>

## Authors

**Jeremiah Gaiser**
School of Information
University of Arizona
Tucson, AZ 85721
[jgaiser@arizona.edu](mailto:jgaiser@arizona.edu)

**Travis J. Wheeler**
College of Pharmacy
University of Arizona
Tucson, AZ 85721
[twheeler@arizona.edu](mailto:twheeler@arizona.edu)
