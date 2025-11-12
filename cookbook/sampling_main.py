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

def uncond_generation_experiment(NUM_STEPS, SEQ_LEN):
    model = ESM3.from_pretrained("esm3-open").to("cuda")
    #random_samples = get_ptm_scores(256, model, 256, sampling_type="random")
    
    top_margin_samples = get_ptm_scores(128, model, SEQ_LEN, sampling_type="top_margin", num_steps=NUM_STEPS)
    margin_mean = np.mean(top_margin_samples)
    
    top_entropy_samples = get_ptm_scores(128, model, SEQ_LEN, sampling_type="entropy", num_steps=NUM_STEPS)
    entropy_mean = np.mean(top_entropy_samples)
    

    return {"margin_mean": margin_mean, "entropy_mean": entropy_mean, "margin_CI": get_confidence_interval(top_margin_samples)}

def experimenter():
    NUM_STEPS_LIST = [8, 16, 32, 64, 128, 256]
    SEQ_LEN = [512, 64]
    for seq_len in SEQ_LEN:
        for num_steps in NUM_STEPS_LIST:
            ret_dict = uncond_generation_experiment(num_steps, seq_len)
            print(f"Seq len={seq_len} and num_steps={num_steps}: margin mean {ret_dict['margin_mean']}, entropy_mean {ret_dict['entropy_mean']}, margin CI {ret_dict['margin_CI']}")




def tertiary_coor_experiment():
    # TODO: finish 
    pass 


if __name__ == "__main__":
    experimenter()
    #uncond_generation_experiment() 