# Low-dimensional statistical manifold embedding of directed graphs (ICLR 2020)
##### by Thorben Funke, Tian Guo, Alen Lancic, and Nino Antulov-Fantulin
Link to publication: [ICLR 2020 Poster](https://iclr.cc/virtual_2020/poster_SkxQp1StDH.html), [arXiv 1905.10227](https://arxiv.org/abs/1905.10227), [OpenReview](https://openreview.net/forum?id=SkxQp1StDH)
## Install requirements 
If you want to execute our code, you should have the following packages installed:

 - tensorflow >=2.0
 - networkx >=2.3
 - numpy >=1.16
 - scikit-learn >= 0.21.3

For Monte-Carlo integration:
 - tensorflow_probability >= 0.7
 
## Executing

 1. As first step, the folder and the configuration needs to be created with the following command, which creates a folder for this configuration and stores the settings in an `.ini`-file.
        
        python create_configuration.py
        
 2. Execute the created configuration with
        
        python execution.py
         
 

Depending on the availability of a GPU, the code will use the specified GPU (default 1) or run on the CPU. 
You may need to set the parameter `-d` or `--dataset`, if the memory usage of the GPU is too high.

During execution a logfile will be created within the created folder.
The results will be saved as `.npz` file (`numpy.load` to access the data) to the subfolder `results`.

Executing the above steps without modifications, will generate the embedding of our toy example of 25 nodes.
To execute different networks, supply the path to a edge list of the same format or adjust the function `read_graph_from_file` of the `generate_data.py` file accordingly. 
(You can use any graph format, which [`NetworkX`](https://networkx.github.io/documentation/stable/reference/readwrite/index.html) supports.)

The network files for the experiments, you can get from the KONECT - The Koblenz Network Collection project.


#### Reference
Funke, T., T. Guo, A. Lancic, and N. Antulov-Fantulin (2020). “Low-dimensional Statistical Manifold Embedding of Directed Graphs.” In: Proceedings of the 8th International Conference on Learning Representations. ICLR’20.


    @inproceedings{funke2019statistical,
	title={Low-dimensional Statistical Manifold Embedding of Directed Graphs},
	author={Funke, Thorben and Guo, Tian and Lancic, Alen and Antulov-Fantulin, Nino},
  	booktitle={Proceedings of the 8th International Conference on Learning Representations},
    series = {ICLR'20},
    year={2020}
    }