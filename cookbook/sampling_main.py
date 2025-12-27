import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain
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
    Randomly place non-overlapping spans.
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

def generate_ligand_prompt(pdb_id, coor_residues):
    '''
    Return ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt), target_inds, mobile_inds, eval_chain
        target_inds: indices of residues in the actual pdb protein chain 
        mobile_inds: indices of residues in the generating prompt
        eval_chain: ProteinChain of the pdb_id 
    '''
    eval_chain = ProteinChain.from_rcsb(pdb_id, "A") # data currently only single chain    

    # first uniformly define length
    #seq_len = random.choice([150,250,350])
    seq_len = random.choice([100])
    seq_prompt_list = list('_' * seq_len)
    structure_prompt = torch.full((seq_len, 37, 3), np.nan)

    # randomly insert residue spans 
    residues = coor_residues.split(" ")
    residues_formatted = [(chr[0], int(chr[1:])) for chr in residues] #List of (Residue letter, position)
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

    target_inds, mobile_inds = [], []

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


def run_tertiary_coordination(pdb_id, coor_residues, model, sampling_type, num_steps, search_type, sample_argmax,\
                              beam_num_child, beam_best_k,beam_explore_best_k,beam_warmup_steps, given_fraction=0.8):
    '''
    Returns (pTM, rMSD) of generated structure. 

    Params:
        eval_residue: (str) Space separated amino acid sequence contianing coordinating residues 
    '''
    #protein_prompt, target_inds, mobile_inds, eval_chain = easier_ligand_prompt(pdb_id, given_fraction=given_fraction)
    protein_prompt, target_inds, mobile_inds, eval_chain = generate_ligand_prompt(pdb_id, coor_residues)
  
    seq_generation_config = GenerationConfig(track="sequence", num_steps=num_steps, temperature=0.7, strategy=sampling_type)
  
    if search_type is not None and search_type in ['mcts', 'beam', 'tree']:
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
        structure_generation_config = GenerationConfig(track="structure", num_steps=num_steps, strategy=sampling_type)
        seq_only_structure_protein_prompt = ESMProtein(sequence=sequence_generation.sequence)
        structure_prediction = model.generate(sequence_generation, structure_generation_config, sample_argmax=sample_argmax) 
    else:
        raise ValueError("Search strategy value either None or mcts, beam, tree")
    gen_ptm = float(structure_prediction.ptm)
    structure_prediction_chain = structure_prediction.to_protein_chain()
    structure_prediction_chain.align(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
    crmsd = structure_prediction_chain.rmsd(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)

    return gen_ptm, crmsd


def get_tertiary_coordination(first_k_ligand, num_gen_per_ligand, model, sampling_type, num_steps, search_type,\
                               beam_num_child=1, beam_best_k=1,beam_warmup_steps=0, beam_explore_num_child=1, sample_argmax=False):
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

        for _ in range(num_gen_per_ligand):
            pTM, rMSD = run_tertiary_coordination(pdb_id, coord_residues, model, sampling_type, num_steps, search_type,\
                            beam_num_child=beam_num_child,beam_explore_num_child=beam_explore_num_child, beam_best_k=beam_best_k,\
                            beam_warmup_steps=beam_warmup_steps, sample_argmax=sample_argmax)
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
    success = 0 
    total_samples=1000
    for _ in range(total_samples):
        protein_prompt, target_inds, mobile_inds, eval_chain = easier_ligand_prompt('7map',given_fraction=0.1)
    
        seq_generation_config = GenerationConfig(track="sequence", num_steps=8, temperature=0.7, strategy='random')
        sequence_generation = model.generate(protein_prompt, seq_generation_config, sample_argmax=False)
        structure_generation_config = GenerationConfig(track="structure", num_steps=8, strategy='random')
        structure_prediction_prompt = ESMProtein(sequence=sequence_generation.sequence)
        structure_prediction = model.generate(structure_prediction_prompt, structure_generation_config, sample_argmax=False)
        gen_ptm = float(structure_prediction.ptm)
        structure_prediction_chain = structure_prediction.to_protein_chain()
        structure_prediction_chain.align(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
        crmsd = structure_prediction_chain.rmsd(eval_chain, mobile_inds=mobile_inds, target_inds=target_inds)
        print(f"PTM {gen_ptm} and RMSD {crmsd}")
        success += int(gen_ptm > 0.8 and crmsd <1.5)
    print(f"Success rate {success}/{total_samples}")

    breakpoint()


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


    

    #exp = Experiment('uncond_ptm', search_type="beam", strategy='random', num_samples=128, baseline_strategies=[], beam_best_k=8, beam_num_child=8)
    #exp.sweep([8], [256], [8])
    
    #result = exp.uncond_ptm(256, 64)
    #print(result)
    #uncond_generation_experiment() 
    #
    #uncond_seq_struct_gen(model, 256, "random", 128)