# Inference time scaling protein language models

TLDR; We investigate test time inference scaling of protein language models under different generation search and sampling strategies. We benchmark on the ESM3 Open model and consider both unconditional structure generation quality (pTM) of various sequence lengths and challenging tertiary coordination protein scaffold design generation.

# TODO: add a plot of figure 4 scaling

# Running experiments
#### Best of N Inference Scaling
To reproduce the best-of-n inference scaling plot results in Figure 1, please run `unconditional_generation_experiment()` in `python cookbook/sampling_main.py`.

#### Tertiary Coodination Protein Scaffold Design 
To generate an unbiased estimate for the tertiary coordination scaffold generation success rate, please run in `python cookbook/sampling_main.py`
'''
model = ESM3.from_pretrained("esm3-open").to("cuda")
get_tertiary_coordination(1, 3500, model, sampling_type='random', num_steps=8, search_type=None):
'''

To calculate the success rates of the beam search strategy as in Figure 4, please run 
'run_tertiary_compute_scaling_experiment()' in `python cookbook/sampling_main.py`

#### Intermediate lookahead value correlation
To reproduce the intermediate denoised lookahead sequences value correlation as in Figure 3, please run in `python cookbook/sampling_main.py`
'''
model = ESM3.from_pretrained("esm3-open").to("cuda")
run_intermediate_sequence_correlation(model, total_samples=156)
''' 







# Code and references
This code is adapted from the ESM3 repository, linked here.
https://github.com/evolutionaryscale/esm. 
You can find more details in their paper: [ESM3](https://www.science.org/doi/10.1126/science.ads0018)

#### ESM3
```
@article {hayes2024simulating,
	author = {Hayes, Thomas and Rao, Roshan and Akin, Halil and Sofroniew, Nicholas J. and Oktay, Deniz and Lin, Zeming and Verkuil, Robert and Tran, Vincent Q. and Deaton, Jonathan and Wiggert, Marius and Badkundri, Rohil and Shafkat, Irhum and Gong, Jun and Derry, Alexander and Molina, Raul S. and Thomas, Neil and Khan, Yousuf A. and Mishra, Chetan and Kim, Carolyn and Bartie, Liam J. and Nemeth, Matthew and Hsu, Patrick D. and Sercu, Tom and Candido, Salvatore and Rives, Alexander},
	title = {Simulating 500 million years of evolution with a language model},
	year = {2025},
	doi = {10.1126/science.ads0018},
	URL = {http://dx.doi.org/10.1126/science.ads0018},
	journal = {Science}
}
```

#### ESM Github (Code / Weights)
```
@software{evolutionaryscale_2024,
  author = {{EvolutionaryScale Team}},
  title = {evolutionaryscale/esm},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14219303},
  URL = {https://doi.org/10.5281/zenodo.14219303}
}
```
