# CS323-2022
Deep Learning for Visual Computing Course @ KAUST


Welcome to the CS 323 - Deep Learning for Visual Computing course! Throughout the course you will learn and implement various deep learning methods with their applications in visual computing. As mentioned in the prerequisites, we expect the students to be familiar with some basic machine learning concepts, linear algebra, multivariate calculus, probability and basic to intermediate programming skills.

## Prerequisites

To get yourselves started and/or refresh your concepts, you can refer to the following links:

**Basic machine learning**: [Basic Intro](https://www.youtube.com/watch?v=ukzFI9rgwfU) and [some basic to intermediate concepts](https://www.coursera.org/learn/machine-learning).

**Linear Algebra**: [Khan Academy](https://www.khanacademy.org/math/linear-algebra)’s linear algebra course provides a good resource to learn/revise the concepts online. Moreover, you can also follow the YouTube Channel [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) for amazing explanations of mathematical concepts including linear algebra.

**Multivariate calculus and Probability**: You can follow Khan Academy videos [here](https://www.khanacademy.org/math/statistics-probability/probability-library) and [here](https://www.khanacademy.org/math/multivariable-calculus).

**Programming**: You should be familiar with [Python](https://www.tutorialspoint.com/python/index.htm). For deep learning, mainly we would be using the [PyTorch](https://pytorch.org/tutorials/) framework. We assume familiarity with Python, but would expect that students are able to study PyTorch on their own to implement the concepts discussed in class. We recommend you make yourselves familiar with the programming frameworks.

Here is a list of some other resources to help you learn about Python and the relevant libraries and packages commonly used in developing deep learning models.

- [A Visual Intro to NumPy and Data Representation](http://jalammar.github.io/visual-numpy/): a visual tutorial to familiarize yourself with ‘numpy’ which is the main library used for scientific computing in Python.
- [Python Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/): a tutorial created for the CS231 Stanford course on CNNs.
- [PyTorch Tensors](https://pytorch.org/tutorials/beginner/pytorch_with_examples.html): a guide to learning about Tensors and forward and backward passes using PyTorch.

## Hardware and Installation
Please note that you need to make sure you have a higher end Nvidia GPU enabled
device for this course. Training/running the code on the CPU becomes impractical
with deep neural networks. You can talk to your academic advisor for the
procurement/availability of such devices and/or refer to cloud services such as
Google Colab.

1. **Google Colab**: Check out the video and helpful article by Google (made inside [Colab](https://colab.research.google.com/notebooks/welcome.ipynb))

The two disadvantages to using Colab are *speed* and *storage*. Colab sessions tend to be slower. Experiments need to be saved to the cloud or manually downloaded before the free 24 hours session is erased and all your local data will be lost.

2. **Local Machines**: We will look at how to setup your local machine and get started on developing deep learning models using [PyTorch](https://pytorch.org/). The most common approach to start developing models that use deep learning in Python is to use the ‘conda’ package and environment manager. For a quick introduction on how to install (and use) ‘conda’ please read the following [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

Once you have ‘conda’ installed, open the newly installed ‘Anaconda prompt’ application on Windows (or a simple terminal on Linux) and run the following
command:
```
conda create --name pytorch pytorch torchvision cudatoolkit=10.0 jupyterlab nb_conda pillow matplotlib scikit-learn scikit-image h5py -c pytorch
```

You can then activate this new environment (called ‘pytorch’) on Windows by running:

```
activate pytorch
```

or

```
conda activate pytorch
```
on Linux. You can now start implementing deep
learning models using PyTorch by simply running:

```
jupyter lab
```

A browser window should automatically show JupyterLab. JupyterLab is a browser based IDE that allows you to create interactive Python ‘notebooks’. Make sure to
select the right kernel type (e.g. ‘Python [conda env:conda-pytorch]’).

To check the installation, run the following inside your notebooks:
```
import torch
torch.cuda.is_available()
```

For a successful installation, it should return True.

## Ibex (Clusters)
Ibex is the name of the clusters at our campus. More information can be found [here](https://www.hpc.kaust.edu.sa/ibex)

A very concise introduction to the usage of Ibex can be found [here](https://www.hpc.kaust.edu.sa/sites/default/files/files/public/Cluster_training/26_11_2018/0_Ibex_cheat_sheet_Nov_26_2018.pdf)

Here, we will introduce the basic functions of Ibex which we will use in this class.

1. To login to the cluster:
```
ssh -X username@glogin.ibex.kaust.edu.sa
```
Change “username” to your portal ID.

2. To use pre-installed applications:
```
module av AppName # view if the app is available
module load AppName # load the app
module purge AppName # unload the app
```

Example:

```
module load miniconda3/4.7.12.1 # load miniconda
```

3. To allocate a node:
It is **very important** to allocate a node before using GPU resources of Ibex.
```
salloc --cpus-per-task=2 --gres=gpu:gtx1080ti:4 --mem=16GB --time=8:00:00 --mail-type ALL
```
In the above command, we want the system to give us a GPU node with 4 GTX1080ti, 2 processors and 16GB memory. It allows us to use this node freely for a duration of 8 hours. You have to estimate the running time of your job before the allocation.
Then you need to login to the allocated node,
```
srun --pty bash -i
```
The command starts an interactive shell which you can operate like in your local machine.
If it is successful, you can check the GPU by nvidia-smi.

4. To run jupyter notebook/lab on Ibex:
Make sure you have an allocated node by using salloc.
```
srun --resv-ports=1 --pty bash -i jupyter notebook --no-browser --ip=0.0.0.0 --port=$SLURM_STEP_RESV_PORTS
```

More information can be found [here](https://www.hpc.kaust.edu.sa/ibex/app?app=jupyter)
