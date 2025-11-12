from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import numpy as np 
import scipy
import math  


def uncond_seq_struct_gen(model, seq_len, sampling_type, num_steps):
    prompt = "_" * seq_len
    protein = ESMProtein(sequence=prompt)

    # generate sequence
    sequence = model.generate(protein, GenerationConfig(track="sequence", num_steps=num_steps, temperature=0.7, strategy=sampling_type))
    structure = model.generate(sequence, GenerationConfig(track="structure", num_steps=num_steps, strategy=sampling_type))

    return structure 

def get_ptm_scores(num_samples, model, seq_len, sampling_type, num_steps):
    ptm_scores = [] 
    for _ in range(num_samples):
        structure = uncond_seq_struct_gen(model, seq_len, sampling_type, num_steps)
        ptm_scores.append(float(structure.ptm))
    return ptm_scores

def get_confidence_interval(values):
    '''
    Return 0.95 low and upper CI 
    '''
    mean_value = np.mean(values)
    std = np.std(values, ddof=1)

    correction = 1.96*std / math.sqrt(len(values))
    return mean_value - correction, mean_value + correction

def main():
    NUM_STEPS=32
    model = ESM3.from_pretrained("esm3-open").to("cuda")
    
    #second_entropy_samples = get_ptm_scores(256, model, 256, sampling_type="entropy")
    #random_samples = get_ptm_scores(256, model, 256, sampling_type="random")
    
    top_margin_samples = get_ptm_scores(128, model, 256, sampling_type="top_margin", num_steps=NUM_STEPS)
    margin_mean = np.mean(top_margin_samples)
    print(f"Margin mean {margin_mean} with CI [{get_confidence_interval(top_margin_samples)}]")
    
    top_entropy_samples = get_ptm_scores(128, model, 256, sampling_type="entropy", num_steps=NUM_STEPS)
    entropy_mean = np.mean(top_entropy_samples)
    print(f"Entropy mean {entropy_mean} with CI [{get_confidence_interval(top_entropy_samples)}]")


if __name__ == "__main__":
    main() 