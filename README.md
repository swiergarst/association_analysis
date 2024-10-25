# Association analysis and survival analysis for the [NCDC usecase 1](https://arxiv.org/abs/2409.01235) project.

## Prerequisites
- [Vantage6](https://docs.vantage6.ai/en/main/index.html) was used to run the association analysis. This repository does not hold the scripts for setting up the vantage6 infrastructure; original analyses were run using the modified vantage6 infrastructure from [here](https://github.com/MaastrichtU-CDS/ncdc-memorabel).
- Additionally, we use a docker image for running the algorithm on the vantage6 node, which can be found [here](https://hub.docker.com/layers/sgarst/association-analysis/1.10/images/sha256-061fd16b100b6a76dfd02d58d46d6ab1894b59e5a71db80037a2a37119e25876?context=repo). It can be pulled by running
```
docker image pull sgarst/association-analysis:1.10
```
(note that this requires docker to be installed, which is also a prerequisite for using vantage6).
- An anaconda environment file is provided, so you can create an environment with all prerequisite python packages by running 
```
conda env create -f environment.yml
```
-  and then activating the environment by running
```
conda activate vantage6
```
## Repository overview
- [results](https://github.com/swiergarst/association_analysis/tree/master/results) holds all association analysis result files (including beta values of linear regression models as well as error rates), and a jupyter notebook to create the table of covariates.
- [survival_analysis](https://github.com/swiergarst/association_analysis/tree/master/survival_analysis) holds the notebooks used for creating the survival analysis results.
- [v6_LinReg_py](https://github.com/swiergarst/association_analysis/tree/master/v6_LinReg_py) holds the source code for the docker image used.
- [run.py](https://github.com/swiergarst/association_analysis/blob/master/run.py)is the main file to run in order to run the association analysis. in order to do so, note that you should modify the lines describing the connection and authentication to the vantage6 server, i.e. modify line 23-24:
```
client = Client("http://localhost", 5000, "/api")
client.authenticate("researcher", "password")`
```
to read the url of your server, as well as use the credentials given to you.
For running different sets of covariates, modify the value for the `model` variable on line 36 to your liking. the outputs will be stored in a json file in the main folder. 
