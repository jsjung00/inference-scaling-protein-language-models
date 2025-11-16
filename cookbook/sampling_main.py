from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
import numpy as np 
import scipy
import math  
import os 
import time 
from datetime import datetime


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


class Experiment:
    def __init__(self, exp_type, strategy='top_margin', baseline_strategies=['entropy', 'random'], num_samples=128, experiment_log_dir="./logs"):
        self.exp_type = exp_type # ['uncond_ptm', 'tertiary_coordination'] #TODO: add diversity?
        self.model = ESM3.from_pretrained("esm3-open").to("cuda")
        self.num_samples = num_samples
        self.experiment_log_dir = experiment_log_dir
        self.strategy = strategy
        self.baseline_strategies = baseline_strategies
        os.makedirs(experiment_log_dir, exist_ok=True)

    def run_experiment(self, seq_len, num_steps):
        if self.exp_type == 'uncond_ptm':
            return self.uncond_ptm(seq_len, num_steps)
        elif self.exp_type == "tertiary_coordination":
            pass 
        else:
            raise ValueError(f"Currently only support exp type ['uncond_ptm', 'tertiary_coordination']")

    def sweep(self, list_num_steps, list_seq_lens):
        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d-%H:%M:%S")
        exp_dir = os.path.join(self.experiment_log_dir, formatted)
        os.makedirs(exp_dir, exist_ok=True)
        log_file_path = os.path.join(exp_dir, "log.txt")
    
        for seq_len in list_num_steps:
            for num_steps in list_num_steps:
                exp_result = self.run_experiment(seq_len, num_steps)
                result_str = f"Seq len={seq_len} and num_steps={num_steps}: {exp_result}\n"
                print(result_str)
                with open(log_file_path, 'a') as f:
                    f.write(result_str)

        print(f"Experiment results finished. Results in {log_file_path}")

    def uncond_ptm(self, seq_len, num_steps):
        results = {}
        strategy_ptm_scores = get_ptm_scores(self.num_samples, self.model, seq_len, num_steps=num_steps, sampling_type=self.strategy)
        strategy_mean_ptm = np.mean(strategy_ptm_scores)
        strategy_ptm_CI = get_confidence_interval(strategy_ptm_scores)
        results[f'{self.strategy}_mean_ptm'] = strategy_mean_ptm
        results[f"{self.strategy}_CI"] = strategy_ptm_CI

        for baseline_strategy in self.baseline_strategies:
            baseline_strategy_ptm_scores = get_ptm_scores(self.num_samples, self.model, seq_len, num_steps=num_steps, sampling_type=baseline_strategy)
            results[f'{baseline_strategy}_mean_ptm'] = np.mean(baseline_strategy_ptm_scores)
        return results 


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
    exp = Experiment('uncond_ptm', num_samples=32)
    result = exp.uncond_ptm(256, 64)
    print(result)
    #uncond_generation_experiment() 