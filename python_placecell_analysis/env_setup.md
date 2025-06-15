# How to run

## Step 1: Creating and activating a Conda Environment

1. Open the Anaconda Navigator or Anaconda Command Prompt.

2. To create a new conda environment, run the following command (replace `myenv` with your desired environment name):

```bash
   conda create --name myenv
```

3. Activate the newly created conda environment by running:

```bash
conda activate myenv
```

You will see the environment name in the prompt, indicating that it's active.

&nbsp;

## Step 2: Installing Dependencies from `requirements.txt`

1. The `requirements.txt` file, lists all the Python packages and their versions required for the analysis.

2. With your conda environment activated, navigate to directory containing the notebook and requirements file using the Anaconda Command Prompt and the `cd` (change directory) command, e.g. 

```bash
cd path\to\my\project
```

3. Install the dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

This will use pip to install the packages listed in requirements.txt into your conda environment.

> If pip isn't installed run the following command and repeat step 3 
> 
> ```bash
> conda install pip
> ```

&nbsp;

## Step 3: Open the jupyter notebook

1. Open jupyter lab or jupyter notebook or open in VScode.

For jupyterlab: 

```bash
jupyter lab 
```

or jupyter notebook

```bash
jupyter notebook 
```

This will open a web browsers and load up jupyter 

2. Open the notebook in the file browser within jupyter lab/notebook 

> jupyterlab is a more advanced version of jupyer notebook and allows for more functionaility.
