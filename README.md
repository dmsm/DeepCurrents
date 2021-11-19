## DeepCurrents | [Webpage](https://people.csail.mit.edu/smirnov/deep-currents/) | [Paper](https://arxiv.org/abs/2111.09383)

<img src="https://people.csail.mit.edu/smirnov/deep-currents/im.png" width="75%" alt="DeepCurrents" />

**DeepCurrents: Learning Implicit Representations of Shapes with Boundaries**<br>
David Palmer*, Dmitriy Smirnov*, Stephanie Wang, Albert Chern, Justin Solomon

### Set-up
To install the neecssary dependencies, run:
```
conda env create -f environment.yml
conda activate DeepCurrents
```

### Training

To prepare the training dataset, first download and extract the FAUST human body meshes:
```
wget -O faust.tar.gz https://www.dropbox.com/s/jgm6hfif6evpi2b/faust.tar.gz?dl=0
tar -xvf faust.tar.gz
```
Then, preprocess the mesh segmentations:
```
./scripts/generate_data.sh
```

To overfit to a single mesh, run:
```
python scripts/train_reconstruction.py --data data/category --idx i --out out_dir
```
You should specify one of `heads`, `torsos`, `arms`, `forearms`, `hands`, or `feet`
as `category`, and indicate an index between 0 and 99 as `i` to pick a mesh from the dataset.

To learn a minimal serfice, run:
```
python scripts/train_minimal.py --boundary boundary_config --idx i --out out_dir
```
Specify the boundary configuration `boundary_config` as either `hopf`, `borromean`, or `trefoil`.

To train a latent model, run:
```
python scripts/train_latent.py --data data/category --out out_dir
```
You should specify one of `heads`, `torsos`, `arms`, `forearms`, `hands`, or `feet`
as `category`.

To monitor the training, launch a TensorBoard instance with `--logdir out_dir`.

### Visualization

To render a turntable GIF from an overfit reconstruction or minimal surface model, run:
```
python scripts/render_current.py --infile out/model/it.pth --outfile out.gif
```
`out/model/it.pth` should be the checkpoint of a trained model.

To render a linear interpolation in boundary or latent space, run:
```
python scripts/render_interpolation.py --infile out/model/it.pth --outfile out.gif --data data/category --interpolation_type interpolation_type
```
`out/model/it.pth` should be the checkpoint of a trained model, and `data/category` the directory to the
dataset used to train the model. You can choose between `latent` or `boundary` as the `interpolation_type`.

### BibTeX
```
@article{palmer2021deepcurrents,
  title={{DeepCurrents}: Learning Implicit Representations of Shapes with Boundaries,
  author={Palmer, David and Smirnov, Dmitriy and Wang, Stephanie and Chern, Albert and Solomon, Justin},
  journal={arXiv:2111.09383},
  year={2021},
}
```
