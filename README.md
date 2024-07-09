
### Step - 0:  Get the code and files

```
git clone https://gitlab.xbiome.com/RavinP/x-sim.git

cd x-sim

```

### Step - 1: Pull the docker image


Login to docker hub and pull docker image locally. You need to have access to [docker hub](harbor-aws.xbiome.com) in order to be able to pull the docker image. If need help with this contact me(ravinp@xbiome.us).

```
docker login harbor-aws.xbiome.com
docker pull harbor-aws.xbiome.com/test/xsim@sha256:f314bb382764019b2b9efc8136467056108ad7c9245b358891e5733c2cc027db
docker tag harbor-aws.xbiome.com/test/xsim@sha256:f314bb382764019b2b9efc8136467056108ad7c9245b358891e5733c2cc027db xsim:latest


```

### Tesitng docker image 

```
docker run -i -t xsim --help 

usage: main.py [-h] --input INPUT --rd2path RD2PATH --rd3_PDBpath RD3_PDBPATH --rd3_SDFpath
               RD3_SDFPATH --type {pdb,sdf} --outdir OUTDIR

Scripts to run X-SMSim

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input SMILES CSV file
  --rd2path RD2PATH, -rd2path RD2PATH
                        Path to 2D Reference Database File (SMILES file)
  --rd3_PDBpath RD3_PDBPATH, -rd3_PDBpath RD3_PDBPATH
                        Path to 3D Reference Database containing PDBs
  --rd3_SDFpath RD3_SDFPATH, -rd3_SDFpath RD3_SDFPATH
                        Path to 3D Reference Database containing SDFs
  --type {pdb,sdf}, -type {pdb,sdf}
                        Ref DB file format type for 3D structures
  --outdir OUTDIR, -o OUTDIR
                        Output Directory
```


### Step - 2: Running a docker version 

```
docker run -v $PWD:/data xsim:latest -i test_input.csv --rd2path DRUGSMIL_smallmol_TTD_tiny.txt --rd3_PDBpath input_data/ref/tiny_MOAD_pdb/ --rd3_SDFpath input_data/ref/tiny_SM_TTD_sdf/ --type pdb --outdir output


docker run -v $PWD:/data xsim:latest -i test_input.csv --rd2path DRUGSMIL_smallmol_TTD_tiny.txt --rd3_PDBpath input_data/ref/tiny_MOAD_pdb/ --rd3_SDFpath input_data/ref/tiny_SM_TTD_sdf/ --type sdf --outdir output


```




### Running with conda env -- old version


```

source ~/anaconda3/etc/profile.d/conda.sh
conda activate /share/home/dhrithi/anaconda3/envs/my-rdkit-env

python main.py -i test_input.csv --rd2path DRUGSMIL_smallmol_TTD_tiny.txt --rd3_PDBpath ../ref/tiny_MOAD_pdb/ --rd3_SDFpath ../ref/tiny_SM_TTD_sdf/ --type pdb --outdir test_outdir
python main.py -i test_input.csv --rd2path DRUGSMIL_smallmol_TTD_tiny.txt --rd3_PDBpath ../ref/tiny_MOAD_pdb/ --rd3_SDFpath ../ref/tiny_SM_TTD_sdf/ --type sdf --outdir test_outdir


```
