import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.generation import generate_structure
import numpy as np 
import scipy
import math  
import os 
import time 
from datetime import datetime
import logging 
from collections import defaultdict
import random 
import torch 
from pathlib import Path 

def uncond_seq_struct_gen(model, seq_len, sampling_type, num_steps, search_type, sample_argmax=False,\
                          beam_num_child=1, beam_best_k=1, beam_warmup_steps=0):
    prompt = "_" * seq_len
    protein = ESMProtein(sequence=prompt)
  
    if search_type is not None and search_type in ['mcts', 'beam', 'tree']:
        # only use search for structure generation for now  
        sequence = model.generate(protein, GenerationConfig(track="sequence", num_steps=num_steps, temperature=0.7, strategy=sampling_type), sample_argmax=sample_argmax)
        structure = model.search_batch_generate([sequence], [GenerationConfig(track="structure", num_steps=num_steps, strategy=sampling_type,\
                                                                beam_num_child=beam_num_child,beam_best_k=beam_best_k, beam_warmup_steps=beam_warmup_steps)],
                                                 sample_argmax=sample_argmax, search_type=search_type)[0] 
    elif search_type is None:
        sequence = model.generate(protein, GenerationConfig(track="sequence", num_steps=num_steps, temperature=0.7, strategy=sampling_type), sample_argmax=sample_argmax)
        structure = model.generate(sequence, GenerationConfig(track="structure", num_steps=num_steps, strategy=sampling_type), sample_argmax=sample_argmax) 
    else:
        raise ValueError("Search strategy value either None or mcts, beam, tree")

    return structure 

def get_ptm_scores(num_samples, model, seq_len, sampling_type, num_steps, search_type, best_of_n=1, beam_num_child=1, beam_best_k=1):
    ptm_scores = [] 
    for _ in range(num_samples):
        best_ptms = []
        for n in range(best_of_n):
            structure = uncond_seq_struct_gen(model, seq_len, sampling_type, num_steps, search_type=search_type, beam_num_child=beam_num_child, beam_best_k=beam_best_k)
            best_ptms.append(float(structure.ptm))

        ptm_scores.append(max(best_ptms))
    return ptm_scores

def place_spans(span_lengths, total_length):
    """
    Randomly place non-overlapping spans. Else, insert in the same location order.

    Returns starting positions for each span. In same order as original span lengths. 
    """
    n_spans = len(span_lengths)
    
    # Shuffle span order for random arrangement
    span_indices = list(range(n_spans))
    random.shuffle(span_indices)
    
    # Greedily place each span in a random valid position
    occupied = []  # list of (start, end) tuples
    positions = [None] * n_spans 
    
    for idx in span_indices:
        span_len = span_lengths[idx]
        
        # Find valid starting positions
        valid_starts = []
        for start in range(total_length - span_len + 1):
            end = start + span_len
            # Check no overlap with occupied
            overlap = any(not (end <= occ_start or start >= occ_end) 
                         for occ_start, occ_end in occupied)
            if not overlap:
                valid_starts.append(start)
        
        # Pick one randomly
        chosen_start = random.choice(valid_starts)
        positions[idx] = chosen_start
        occupied.append((chosen_start, chosen_start + span_len))
    
    return positions

def generate_ligand_prompt(pdb_id, coor_residues, chain_id='A', place_orig_order=False):
    '''
    Generates a ligand prompt as in the challenging tertiary coordination task in the ESM3 paper 

    Return ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt), target_inds, mobile_inds, eval_chain
        target_inds: indices of residues in the actual pdb protein chain 
        mobile_inds: indices of residues in the generating prompt
        eval_chain: ProteinChain of the pdb_id 

    Params:
        place_orig_order: (bool) If true just mask original protein chain residues not in coor_residues
        
        use_orig_len: (bool) If true set scaffold sequence length to original chain length
        shuffle_spans: (bool) If true shuffle the span insertion location as in the original task 
    '''
    eval_chain = ProteinChain.from_rcsb(pdb_id, chain_id) # data currently only single chain    

    # first uniformly define length
    if place_orig_order:
        seq_len = len(eval_chain)
    else:
        seq_len = random.choice([150,250,350])

    seq_prompt_list = list('_' * seq_len)
    structure_prompt = torch.full((seq_len, 37, 3), np.nan).float()
    target_inds, mobile_inds = [], []
    residues = coor_residues.split(" ")
    residues_formatted = [(chr[0], int(chr[1:])) for chr in residues] #List of (Residue letter, position)

    if place_orig_order:
        for (letter, position) in residues_formatted:
            prompt_idx = eval_chain.residue_index.tolist().index(position)
            mobile_inds.append(prompt_idx)
            seq_prompt_list[prompt_idx] = letter 
            assert seq_prompt_list[prompt_idx] == eval_chain.sequence[prompt_idx]
        target_inds = mobile_inds
        motif = eval_chain[np.array(mobile_inds)]
        motif_atom37_positions = motif.atom37_positions 
        structure_prompt[np.array(mobile_inds)] = torch.tensor(motif_atom37_positions).float()
        seq_prompt = ''.join(seq_prompt_list)
        protein_prompt = ESMProtein(sequence=seq_prompt, coordinates=structure_prompt)
        return protein_prompt, target_inds, mobile_inds, eval_chain

    # randomly insert residue spans 
    residue_spans = []
    current_span = [residues_formatted[0]]

    for i in range(1, len(residues_formatted)):
        letter, position = residues_formatted[i]
        prev_letter, prev_position = residues_formatted[i-1]

        if (position - prev_position) <= 5: 
            current_span.append((letter, position))
        else:
            residue_spans.append(current_span)
            current_span = [(letter, position)]

    if len(current_span) > 0: residue_spans.append(current_span)
 
    span_lengths = [(span[-1][1]-span[0][1]+1) for span in residue_spans]
    span_start_indices = place_spans(span_lengths, seq_len)
    

    for i in range(len(residue_spans)):
        span = residue_spans[i]
        span_start_idx = span_start_indices[i]
        shift = span_start_idx - span[0][1]
        for (letter, position) in span:
            seq_prompt_list[shift + position] = letter 
            list_idx = eval_chain.residue_index.tolist().index(position)
            target_inds.append(list_idx) #index in the actual pdb protein chain 
            mobile_inds.append(shift+position) #index in the prompt sequence

            assert eval_chain.sequence[list_idx] == letter 
            coord = eval_chain.atom37_positions[list_idx] #(37,3)
            structure_prompt[shift+position] = torch.tensor(coord)

    seq_prompt = ''.join(seq_prompt_list)
    
    protein_prompt = ESMProtein(sequence=seq_prompt, coordinates=structure_prompt)
    return protein_prompt, target_inds, mobile_inds, eval_chain

def easier_ligand_prompt(pdb_id, chain_id='A', motif_inds=None, given_fraction=0.75):
    """Contiguous motif scaffolding where motif is in the same location as the original protein"""
    eval_chain = ProteinChain.from_rcsb(pdb_id, chain_id) # data currently only single chain    
    
    seq_len = len(list(eval_chain.sequence))
    sequence_prompt = np.array(["_"] * seq_len)

    if motif_inds is None:
        num_givens = int(seq_len * given_fraction)
        start_idx = np.random.choice(np.arange(0, seq_len-num_givens))
        motif_inds = np.arange(start_idx, start_idx+num_givens)

    motif_seq = eval_chain[motif_inds].sequence   
    motif_atom37_positions = eval_chain[motif_inds].atom37_positions 
    
    sequence_prompt[motif_inds] = list(motif_seq)
    structure_prompt = torch.full((seq_len, 37, 3), np.nan).float()
    structure_prompt[motif_inds] = torch.tensor(motif_atom37_positions).float() 

    sequence_prompt = ''.join(list(sequence_prompt))
    mobile_inds, target_inds = list(motif_inds), list(motif_inds)

    protein_prompt = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
    return protein_prompt, target_inds, mobile_inds, eval_chain

def medium_ligand_prompt(pdb_id, chain_id='A', motif_inds=None, given_fraction=0.75):
    """
    Contiguous motif scaffolding where contiguous block motif is inserted randomly into starting partially masked sequence
        Randomly choose a block from the chain to define the motif
        Randomly choose an insert location in the prompt sequence
        Define the prompt sequence to be either length 150, 250, or 350
    """
    eval_chain = ProteinChain.from_rcsb(pdb_id, chain_id) # data currently only single chain    
    prompt_seq_len = random.choice([150,250,350])
    sequence_prompt = np.array(["_"] * prompt_seq_len)
    motif_seq_len = len(eval_chain) 

    if motif_inds is None:
        num_givens = int(min(prompt_seq_len, motif_seq_len) * given_fraction)
        motif_start_idx = np.random.choice(np.arange(0, motif_seq_len-num_givens))
        motif_inds = np.arange(motif_start_idx, motif_start_idx+num_givens)    
        insert_start_idx = np.random.choice(np.arange(0, prompt_seq_len-num_givens))
        insert_inds = np.arange(insert_start_idx, insert_start_idx+num_givens)

    motif_seq = eval_chain[motif_inds].sequence   
    motif_atom37_positions = eval_chain[motif_inds].atom37_positions 
    
    sequence_prompt[insert_inds] = list(motif_seq)
    structure_prompt = torch.full((prompt_seq_len, 37, 3), np.nan).float()
    structure_prompt[insert_inds] = torch.tensor(motif_atom37_positions).float() 

    sequence_prompt = ''.join(list(sequence_prompt))
    mobile_inds, target_inds = [int(elm) for elm in insert_inds], [int(elm) for elm in motif_inds]
    protein_prompt = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
    return protein_prompt, target_inds, mobile_inds, eval_chain


def run_tertiary_coordination(pdb_id, coor_residues, model, sampling_type, num_steps, search_type, sample_argmax,\
                              beam_num_child, beam_best_k,beam_explore_best_k,beam_warmup_steps):
    '''
    Returns (pTM, rMSD) of generated structure. 

    Params:
        eval_residue: (str) Space separated amino acid sequence contianing coordinating residues 
    '''
    #protein_prompt, target_inds, mobile_inds, eval_chain = easier_ligand_prompt(pdb_id, given_fraction=given_fraction)
    protein_prompt, target_inds, mobile_inds, eval_chain = generate_ligand_prompt(pdb_id, coor_residues, place_orig_order=True)
  
    seq_generation_config = GenerationConfig(track="sequence", num_steps=num_steps, temperature=0.7, strategy=sampling_type)
  
    if search_type is not None and search_type in ['mcts', 'beam', 'tree']:
        # TODO: FIX 
        raise ValueError("Search should be on sequence!!")
        # only use search for structure generation for now  
        sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=sample_argmax) 
        structure_generation_config = GenerationConfig(track="structure", num_steps=num_steps, strategy=sampling_type,\
                                        beam_num_child=beam_num_child,\
                                            beam_best_k=beam_best_k,beam_explore_best_k=beam_explore_best_k, beam_warmup_steps=beam_warmup_steps)
        seq_only_structure_protein_prompt = ESMProtein(sequence=sequence_generation.sequence)
        structure_prediction = model.search_batch_generate([sequence_generation], [structure_generation_config],
                                                 sample_argmax=sample_argmax, search_type=search_type)[0] 
    elif search_type is None:
        sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=sample_argmax)
        structure_prediction = generate_structure(sequence_generation, model, best_of_n=16, batched_generate=False) 
    else:
        raise ValueError("Search strategy value either None or mcts, beam, tree")
    gen_ptm = float(structure_prediction.ptm)
    structure_prediction_chain = structure_prediction.to_protein_chain()
    structure_prediction_chain.align(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
    crmsd = structure_prediction_chain.rmsd(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)

    return gen_ptm, crmsd


def get_tertiary_coordination(first_k_ligand, num_gen_per_ligand, model, sampling_type, num_steps, search_type,\
                               beam_num_child=1, beam_best_k=1,beam_warmup_steps=0, beam_explore_best_k=1, sample_argmax=False):
    '''
    Returns total sucesss rate across all the k ligands. (total passed / total generated)

    Params:
        first_k_ligand: (int) Evaluate on the first k ligands in the document  
        num_gen_per_ligand: (int) Number of generations to do per ligand
    '''
    ligand_generation_stats = defaultdict(list) #key: pdb_id, value: [(pTM, rMSD), (pTM, rMSD), ...]

    parent_folder = Path(__file__).resolve().parent
    coord_path = parent_folder / "coord_residues.txt"
    with open(coord_path, 'r') as fp:
        coord_residues_data = fp.readlines()

    eval_residue_data = coord_residues_data[:first_k_ligand] #list of space seperated string with | separating PDB id w/ residues 
    for k in range(first_k_ligand):
        eval_residue_line = eval_residue_data[k]
        pdb_id, coord_residues = eval_residue_line.split("|")[0], eval_residue_line.split("|")[1]  
        pdb_id, coord_residues = pdb_id.strip(), coord_residues.strip()
        success = 0
        for i in range(num_gen_per_ligand):
            pTM, rMSD = run_tertiary_coordination(pdb_id, coord_residues, model, sampling_type, num_steps, search_type,\
                            beam_num_child=beam_num_child, beam_best_k=beam_best_k, beam_explore_best_k=beam_explore_best_k, 
                            beam_warmup_steps=beam_warmup_steps, sample_argmax=sample_argmax)
            print(f"Ligand {pdb_id} PTM {pTM} rMSD {rMSD}")
            success += int(pTM > 0.8 and rMSD < 1.5)
            print(f"Ligand {pdb_id} success rate so far {success} / {i+1}")
            ligand_generation_stats[pdb_id].append((pTM, rMSD))

    # calculate the total success rate where succeed if pTM > 0.8 and rMSD < 1.5 
    total_samples, total_success = 0, 0
    ptms, rmsds = [], [] 
    best_ptm, best_rmsd = 0, 1e10
    for ligand,results in ligand_generation_stats.items():
        total_samples += len(results)
        success_samples = list(filter(lambda x: (x[0] > 0.8) and (x[1] < 1.5), results))
        total_success += len(success_samples)
        for x in results:
            best_ptm = max(best_ptm, x[0])
            best_rmsd = min(best_rmsd, x[1])
            ptms.append(x[0])
            rmsds.append(x[1])

    # print the best ptm and best rMSD
    print(f"Best PTM {best_ptm} and rsmd {best_rmsd}")
    success_rate = total_success/total_samples

    return success_rate, ptms, rmsds


def get_confidence_interval(values):
    '''
    Return 0.95 low and upper CI 
    '''
    mean_value = np.mean(values)
    std = np.std(values, ddof=1)

    correction = 1.96*std / math.sqrt(len(values))
    return mean_value - correction, mean_value + correction


class Experiment:
    def __init__(self, exp_type, search_type=None, strategy='top_margin', baseline_strategies=['entropy', 'random'],\
                    num_samples=128, experiment_log_dir="./logs", beam_num_child=1, beam_best_k=1):
        self.exp_type = exp_type # ['uncond_ptm', 'tertiary_coordination'] #TODO: add diversity?
        self.model = ESM3.from_pretrained("esm3-open").to("cuda")
        self.num_samples = num_samples
        self.experiment_log_dir = experiment_log_dir
        self.strategy = strategy
        self.search_type = search_type
        self.baseline_strategies = baseline_strategies
        self.beam_num_child = beam_num_child
        self.beam_best_k = beam_best_k
        os.makedirs(experiment_log_dir, exist_ok=True)

    def run_experiment(self, seq_len, num_steps, best_of_n):
        if self.exp_type == 'uncond_ptm':
            return self.uncond_ptm(seq_len, num_steps, best_of_n)
        elif self.exp_type == "tertiary_coordination":
            pass 
        else:
            raise ValueError(f"Currently only support exp type ['uncond_ptm', 'tertiary_coordination']")

    def sweep(self, list_num_steps, list_seq_lens, list_nfes):
        now = datetime.now()
        formatted = now.strftime("%Y-%m-%d-%H:%M:%S")
        exp_dir = os.path.join(self.experiment_log_dir, formatted)
        os.makedirs(exp_dir, exist_ok=True)
        log_file_path = os.path.join(exp_dir, "log.txt")
    
        for seq_len in list_seq_lens:
            for num_steps in list_num_steps:
                for nfe in list_nfes:
                    best_of_n = nfe // num_steps
                    if best_of_n < 1: continue 
                    if self.search_type is not None:
                        best_of_n = 1
                    exp_result = self.run_experiment(seq_len, num_steps, best_of_n)
                    result_str = f"Seq len={seq_len} and num_steps={num_steps} and bestofn={best_of_n} search_type {self.search_type} beam_num_child={self.beam_num_child} beam_best_k={self.beam_best_k} : {exp_result}\n"
                    print(result_str)
                    with open(log_file_path, 'a') as f:
                        f.write(result_str)

        print(f"Experiment results finished. Results in {log_file_path}")

    def uncond_ptm(self, seq_len, num_steps, best_of_n):
        results = {}
        strategy_ptm_scores = get_ptm_scores(self.num_samples, self.model, seq_len,\
                                            num_steps=num_steps, sampling_type=self.strategy,\
                                                best_of_n=best_of_n, search_type=self.search_type,\
                                                beam_best_k=self.beam_best_k, beam_num_child=self.beam_num_child)
        strategy_mean_ptm = np.mean(strategy_ptm_scores)
        strategy_ptm_CI = get_confidence_interval(strategy_ptm_scores)
        results[f'{self.strategy}_mean_ptm'] = strategy_mean_ptm
        results[f"{self.strategy}_CI"] = strategy_ptm_CI

        for baseline_strategy in self.baseline_strategies:
            baseline_strategy_ptm_scores = get_ptm_scores(self.num_samples, self.model, seq_len, num_steps=num_steps,
                                                         sampling_type=baseline_strategy, best_of_n=best_of_n, search_type=self.search_type)
            results[f'{baseline_strategy}_mean_ptm'] = np.mean(baseline_strategy_ptm_scores)
            results[f"{baseline_strategy}_CI"] = get_confidence_interval(baseline_strategy_ptm_scores)
        return results 
    
    def tertiary_coordination(self, seq_len, num_steps):
        results = {} 

        pass 

def get_fold_std(sequence_obj, model, best_of_n=[8,32,128], num_samples=128, sample_argmax=False):
    '''
    Get the std and mean of best-of-n generated fold score 
    '''
    best_of_n_score_std = {k: None for k in best_of_n} #key: n in best-of-n, value: variance of best-of-n score
    best_of_n_score_mean = {k: None for k in best_of_n}
    for n in best_of_n:
        best_fold_scores = []
        for _ in range(num_samples):
            best_structure = generate_structure(sequence_obj, model, best_of_n=n, sample_argmax=sample_argmax)
            best_of_n_ptm = float(best_structure.ptm)
            best_fold_scores.append(best_of_n_ptm)
        best_fold_scores = np.array(best_fold_scores)
        mean, std = np.mean(best_fold_scores), np.std(best_fold_scores)
        best_of_n_score_mean[n] = mean 
        best_of_n_score_std[n] = std 
        print(f"BestofN={n} Mean: {mean}, Std: {std}")
    
    return best_of_n_score_std, best_of_n_score_mean


### misc experiment drivers ###
def run_fold_std_experiment(pdb_id='7map'):
    protein_prompt, target_inds, mobile_inds, eval_chain = medium_ligand_prompt(pdb_id, given_fraction=0.25)
    seq_generation_config = GenerationConfig(track="sequence", num_steps=8, temperature=0.7, strategy='random')
    sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=False)
    best_of_n_score_std, best_of_n_score_mean = get_fold_std(sequence_generation, model, best_of_n=[16])
    print(best_of_n_score_std)
    print(best_of_n_score_mean)

def run_best_of_n_experiment(model, total_samples=1000, batched_generate=True):
    '''
    Pass 
    '''
    success = 0 
    ptms, rmsds = [], [] 
    for i in range(total_samples):
        protein_prompt, target_inds, mobile_inds, eval_chain = medium_ligand_prompt('7map',given_fraction=0.25)
        seq_generation_config = GenerationConfig(track="sequence", num_steps=8, temperature=0.7, strategy='random')
        sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=False)
        structure_prediction = generate_structure(sequence_generation, model, best_of_n=16, batched_generate=batched_generate)
        
        gen_ptm = float(structure_prediction.ptm)
        structure_prediction_chain = structure_prediction.to_protein_chain()
        structure_prediction_chain.align(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
        crmsd = structure_prediction_chain.rmsd(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
        print(f"sample {i}: PTM {gen_ptm} and RMSD {crmsd}")
        success += int(gen_ptm > 0.8 and crmsd <1.5)
        ptms.append(gen_ptm)
        rmsds.append(crmsd)

    print(f"PTM mean {np.mean(ptms)} and std {np.std(ptms)} and RSMD mean {np.mean(rmsds)} and std {np.std(rmsds)}")
    print(f"Success rate {success}/{total_samples}")


def run_given_fraction_sweep(model, given_fractions=[0.05, 0.1, 0.25], total_samples=32, pdb_id='7map'):
    '''
    Experiment to determine what a reasonable given fraction is to make the medium ligand prompt scaffold generation problem 
        reasonably difficult. 
    '''
    success_rate_per_fraction = defaultdict(float)
    for given_fraction in given_fractions:
        success = 0 
        for i in range(total_samples):
            protein_prompt, target_inds, mobile_inds, eval_chain = medium_ligand_prompt(pdb_id, given_fraction=given_fraction)
            seq_generation_config = GenerationConfig(track="sequence", num_steps=8, temperature=0.7, strategy='random')
            sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=False)
            structure_prediction = generate_structure(sequence_generation, model, best_of_n=16)
            gen_ptm = float(structure_prediction.ptm)
            structure_prediction_chain = structure_prediction.to_protein_chain()
            structure_prediction_chain.align(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
            crmsd = structure_prediction_chain.rmsd(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
            print(f"PTM {gen_ptm} and RMSD {crmsd}")
            success += int(gen_ptm > 0.8 and crmsd <1.5)
            print(f"Givenfraction={given_fraction}. Success rate so far {success} / {i+1}")

        success_rate = success/total_samples
        success_rate_per_fraction[given_fraction] = success_rate

        print(f"Given fraction={given_fraction} has success rate {success} / {total_samples}")
    return success_rate_per_fraction

def run_orig_order_tertiary_experiment(model):
    success = 0 
    total_samples=10000
    for _ in range(total_samples):
        protein_prompt, target_inds, mobile_inds, eval_chain = medium_ligand_prompt('7map',given_fraction=0.25)
        seq_generation_config = GenerationConfig(track="sequence", num_steps=8, temperature=0.7, strategy='random')
        sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=False)
        structure_prediction = generate_structure(sequence_generation, model, best_of_n=8)
        
        gen_ptm = float(structure_prediction.ptm)
        structure_prediction_chain = structure_prediction.to_protein_chain()
        structure_prediction_chain.align(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
        crmsd = structure_prediction_chain.rmsd(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
        print(f"PTM {gen_ptm} and RMSD {crmsd}")
        success += int(gen_ptm > 0.8 and crmsd <1.5)
    print(f"Success rate {success}/{total_samples}")
    pass 

def run_intermediate_sequence_correlation(model, pdb_id='8GXP', coor_residues='W317 C320 A321 H323 V376 F377 L396 I400 H479 Y502',
                                           total_samples=1000, correlation_file_name='correlation.txt'):
    '''
    Calculates the correlation between the intermediate sequence best-of-n fold score and the final generated sequence best-of-n fold score
    Use the tertiary coordination "generate_ligand_prompt" with place_in_order version to get the seqeunce prompt
    '''
    current_file_directory = str(Path(__file__).parent)
    save_file_path = os.path.join(current_file_directory, correlation_file_name)
    logger = logging.getLogger("correlation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(save_file_path)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)

    for _ in range(total_samples):
        protein_prompt, target_inds, mobile_inds, eval_chain = generate_ligand_prompt(pdb_id, coor_residues, place_orig_order=True)
        seq_generation_config = GenerationConfig(track="sequence", num_steps=8, temperature=0.7, strategy='random', run_intermediate_correlation=True,\
                                                 correlation_data_file_path=save_file_path)
        sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=False)
     
     

if __name__ == "__main__":
    logging.basicConfig(
        filename="experiment_beam_tertiary.log",
        level=logging.INFO,
        format="%(message)s"
    )
    model = ESM3.from_pretrained("esm3-open").to("cuda")
    '''
    for (beam_best_k, beam_num_child) in [(4, 8), (6, 12), (10, 32), (12, 48), (16, 64), (20, 100)]:
        success_rate, ptms, rmsds = get_tertiary_coordination(2, 128, model, sampling_type='random', num_steps=8, beam_best_k=beam_best_k, beam_num_child=beam_num_child, search_type='beam')
        ptms_percentiles = np.percentile(ptms, [50,90,99])
        rmsds_percentiles = np.percentile(rmsds, [1,10,50])
        logging.info(f"First 5 Ligand, 128 generations each. Beam_k={beam_best_k} Num_child={beam_num_child}: success={success_rate} ptm_50/90/99={ptms_percentiles} rmsds_1/10/50={rmsds_percentiles}")
    '''
    run_intermediate_sequence_correlation(model)
    #get_tertiary_coordination(2, 4096, model, sampling_type='random', num_steps=8,\
    #                          search_type=None)
    
    breakpoint()
    
    protein_prompt, target_inds, mobile_inds, eval_chain = generate_ligand_prompt('8GXP', 'W317 C320 A321 H323 V376 F377 L396 I400 H479 Y502', place_orig_order=True)
    seq_generation_config = GenerationConfig(track="sequence", num_steps=8, temperature=0.7, strategy='random')
    sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=False)
    generate_structure(sequence_generation, model, best_of_n=16, sample_argmax=False)
    
    #run_given_fraction_sweep(model, given_fractions=[0.25], total_samples=128)
    #

    # next todo: figure out what is a fair n to have the maximum fold score be representative. Hoping 8 is fine
    # then, run the best of N code below. you can refactor this pretty easily. you should also for loop over PDB ids.  

    '''
    for (beam_best_k, beam_num_child, beam_explore_best_k) in [(32, 1, 32)]:
        max_ptm,min_rmsd, ptms, rmsds = 0,1e10, [], []
        for i in range(1000000):
            ptm, rmsd = run_tertiary_coordination('7map', 'D25 G27 A28 D29 D30 G48 G49 V50', model, sampling_type='random',\
                                    num_steps=64, search_type=None, sample_argmax=False, beam_num_child=beam_num_child,\
                                    beam_best_k=beam_best_k,beam_explore_best_k=beam_explore_best_k,\
                                    beam_warmup_steps=7, given_fraction=0.5)
            #structure = uncond_seq_struct_gen(model, 269, sampling_type='random', num_steps=8, search_type=None, sample_argmax=False,\
        #                   beam_num_child=1, beam_best_k=1)
            
            max_ptm = max(max_ptm, ptm)
            min_rmsd = min(min_rmsd, rmsd)
            print(f"generation{i} Max PTM {max_ptm} and min RMSD {min_rmsd}")
            ptms.append(ptm)
            rmsds.append(rmsd)
    
        both_ptm_rmsd = sorted(list(zip(ptms, rmsds)), key=lambda x: x[0], reverse=True)

        ptms_percentiles = np.percentile(ptms, [50,90,99])
        rmsds_percentiles = np.percentile(rmsds, [1,10,50])
        logging.info(f"7map, 1mil generations 64 step. Beam_k={beam_best_k} Num_child={beam_num_child}. MaxPTM: {max_ptm}")
        logging.info(f"7map, 1mil generations 64 step. Beam_k={beam_best_k} Num_child={beam_num_child}. Ptm50/90/99: {ptms_percentiles}")
        logging.info(f"7map, 1mil generations 64 step. Beam_k={beam_best_k} Num_child={beam_num_child}. RMSD1/10/50: {rmsds_percentiles}")
        logging.info(f"7map, 1mil generations 64 step. Beam_k={beam_best_k} Num_child={beam_num_child}. Top10ptms: {both_ptm_rmsd[:10]}") 
    '''

    #exp = Experiment('uncond_ptm', search_type="beam", strategy='random', num_samples=128, baseline_strategies=[], beam_best_k=8, beam_num_child=8)
    #exp.sweep([8], [256], [8])
    
    #result = exp.uncond_ptm(256, 64)
    #print(result)
    #uncond_generation_experiment() 
    #
    #uncond_seq_struct_gen(model, 256, "random", 128)