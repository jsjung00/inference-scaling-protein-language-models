import re
from scipy.stats import pearsonr

def get_correlation_from_file(file_path):
    """
    Calculate the correlation between scores at timesteps 0-6 and the final score at timestep 7.
    
    Args:
        file_path: Path to the text file containing timestep scores
        
    Returns:
        Dictionary mapping each timestep (0-6) to its Pearson correlation with timestep 7
    """
    # Parse the file and extract scores
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract timestep and score from each line
    pattern = r'Timestep (\d+): ptmScore ([\d.]+) \| rmsdScore ([\d.]+)'
    
    # Group scores by generation (each generation has timesteps 0-7)
    ptm_generations, rmsd_generations = [], []
    current_ptm_generation, current_rmsd_generation = {}, {}
    
    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            timestep = int(match.group(1))
            ptmScore = float(match.group(2))
            rmsdScore = float(match.group(3))
            
            # If we see timestep 0 and current_generation is not empty, start a new generation
            if timestep == 0 and current_ptm_generation and current_rmsd_generation:
                ptm_generations.append(current_ptm_generation)
                rmsd_generations.append(current_rmsd_generation)
                current_ptm_generation, current_rmsd_generation = {}, {}
            
            current_ptm_generation[timestep] = ptmScore
            current_rmsd_generation[timestep] = rmsdScore
    
    # Don't forget the last generation
    if current_ptm_generation and current_rmsd_generation:
        ptm_generations.append(current_ptm_generation)
        rmsd_generations.append(current_rmsd_generation)
    
    # Organize scores by timestep across all generations
    ptm_scores_by_timestep, rmsd_scores_by_timestep = {i: [] for i in range(8)}, {i: [] for i in range(8)}
    
    for i in range(len(ptm_generations)):
        ptm_gen = ptm_generations[i]
        rmsd_gen = rmsd_generations[i]
        for timestep in range(8):
            if timestep in ptm_gen:
                ptm_scores_by_timestep[timestep].append(ptm_gen[timestep])
            if timestep in rmsd_gen:
                rmsd_scores_by_timestep[timestep].append(rmsd_gen[timestep])
            
    
    # Calculate correlation of each timestep (0-6) with timestep 7
    ptm_correlations, rmsd_correlations = {}, {}
    ptm_timestep_7_scores, rmsd_timestep_7_scores = ptm_scores_by_timestep[7], rmsd_scores_by_timestep[7]
    
    for timestep in range(7):
        ptm_timestep_scores = ptm_scores_by_timestep[timestep]
        rmsd_timestep_scores = rmsd_scores_by_timestep[timestep]

        if len(ptm_timestep_scores) > 1:
            corr, p_value = pearsonr(ptm_timestep_scores, ptm_timestep_7_scores)
            ptm_correlations[timestep] = {
                'correlation': corr,
                'p_value': p_value,
                'n_samples': len(ptm_timestep_scores)
            }

        if len(rmsd_timestep_scores) > 1:
            corr, p_value = pearsonr(rmsd_timestep_scores, rmsd_timestep_7_scores)
            rmsd_correlations[timestep] = {
                'correlation': corr,
                'p_value': p_value,
                'n_samples': len(rmsd_timestep_scores)
            }

    return ptm_correlations, rmsd_correlations



if __name__ == "__main__":
    ptm_correlations,rmsd_correlations = get_correlation_from_file('/home/jsjung00/Desktop/Code/esm-sampling/cookbook/full_correlation.txt')
    print(ptm_correlations)
    print(rmsd_correlations)
    breakpoint()



