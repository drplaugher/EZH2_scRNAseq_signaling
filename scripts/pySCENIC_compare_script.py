"""
pySCENIC Cross-Treatment Comparison Script

Analyzes and compares regulon activity and networks across different treatment groups.
Provides comprehensive cross-treatment analysis including regulon overlap, activity patterns,
target gene comparison, and treatment similarity assessment.

Author: Daniel Plaugher
Created: 6-7-25, cleaned with help of Claude
"""

# %% Libraries
# System and file operations
import os
import pickle
import glob
import re
from collections import defaultdict, Counter

# Core data processing libraries
import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from adjustText import adjust_text
import networkx as nx
from statannot import add_stat_annotation

# %% Initial setup
# =============================================================================
# CONFIGURATION - MODIFY THESE SECTIONS FOR YOUR DATA
# =============================================================================

# Define treatment groups
treatment_groups = [
    "1_Jun-Lung_Control",
    "2_Jun-Lung_GSK",
   # "3_Aug-Lung2_PD1",
   "6_Oct-Lung_Placebo",
    "4_Aug-Lung3_GSK"
   # "5_Aug-Lung4_Combo",  
]

# For better labeling in plots
treatment_labels = {
    "1_Jun-Lung_Control": "Control",
    "2_Jun-Lung_GSK": "GSK_June",
    #"3_Aug-Lung2_PD1": "PD1",
    "4_Aug-Lung3_GSK": "GSK_Aug",
   # "5_Aug-Lung4_Combo": "Combo",
    "6_Oct-Lung_Placebo": "Placebo"
}

# Create color mapping for treatment groups
treatment_colors = {
    "1_Jun-Lung_Control": "#000000",   # 
    "2_Jun-Lung_GSK": "#558ED5",       # blue
   # "3_Aug-Lung2_PD1": "#FF66CC",      # pink
    "4_Aug-Lung3_GSK": "#95B3D7",      # light blue
   # "5_Aug-Lung4_Combo": "#7030A0",    # purple
    "6_Oct-Lung_Placebo": "#595959"    # black
}

# Define paths
BASE_DIR = "C:/Users/Daniel/Documents/TD_data/TF inference/pySCENIC"

# Define paths - MODIFY THESE FOR YOUR DIRECTORY STRUCTURE
#BASE_DIR = "DEFINE_YOUR_BASE_DIRECTORY"  # e.g., "/path/to/pySCENIC"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
COMPARISON_DIR = os.path.join(BASE_DIR, "comparison_results-6-7-25")
os.makedirs(COMPARISON_DIR, exist_ok=True)

# %% Functions for data loading
def load_auc_scores(treatment_group):
    """Load AUC scores for a given treatment group"""
    file_path = os.path.join(RESULTS_DIR, treatment_group, f"auc_scores_{treatment_group}.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)
    else:
        print(f"AUC scores file not found for {treatment_group}")
        return None

def load_rss_scores(treatment_group):
    """Load RSS scores (Regulon Specificity Scores) for a given treatment group"""
    file_path = os.path.join(RESULTS_DIR, treatment_group, "further_analysis", "regulon_specificity_scores.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)
    else:
        print(f"RSS scores file not found for {treatment_group}")
        return None

def load_celltype_avg_auc(treatment_group):
    """Load cell type average AUC scores for a given treatment group"""
    file_path = os.path.join(RESULTS_DIR, treatment_group, "further_analysis", "celltype_avg_auc.csv")
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0)
    else:
        print(f"Cell type average AUC file not found for {treatment_group}")
        return None

def load_regulons(treatment_group):
    """Load regulon objects for a given treatment group"""
    file_path = os.path.join(RESULTS_DIR, treatment_group, "regulons.p")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        print(f"Regulons file not found for {treatment_group}")
        return None

def load_all_regulon_names():
    """Load all unique regulon names across all treatment groups"""
    all_regulons = set()
    for treatment_group in treatment_groups:
        auc_scores = load_auc_scores(treatment_group)
        if auc_scores is not None:
            all_regulons.update(auc_scores.columns)
    return sorted(list(all_regulons))

def make_valid_filename(s):
    """Convert a string to a valid filename"""
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)

# %% 1. Load and organize data across treatments
print("Loading data from all treatment groups...")

# Create data structures to hold results
auc_scores_dict = {}
rss_scores_dict = {}
celltype_avg_auc_dict = {}
regulons_dict = {}

# Load data for each treatment group
for treatment_group in treatment_groups:
    print(f"Loading data for {treatment_group}...")
    
    # Load AUC scores
    auc_scores = load_auc_scores(treatment_group)
    if auc_scores is not None:
        auc_scores_dict[treatment_group] = auc_scores
        print(f"  Loaded AUC scores: {auc_scores.shape}")
    
    # Load RSS scores
    rss_scores = load_rss_scores(treatment_group)
    if rss_scores is not None:
        rss_scores_dict[treatment_group] = rss_scores
        print(f"  Loaded RSS scores: {rss_scores.shape}")
    
    # Load cell type average AUC
    celltype_avg_auc = load_celltype_avg_auc(treatment_group)
    if celltype_avg_auc is not None:
        celltype_avg_auc_dict[treatment_group] = celltype_avg_auc
        print(f"  Loaded cell type average AUC: {celltype_avg_auc.shape}")
    
    # Load regulons
    regulons = load_regulons(treatment_group)
    if regulons is not None:
        regulons_dict[treatment_group] = regulons
        print(f"  Loaded {len(regulons)} regulons")

print("\nData loading complete!\n")

# %% 2. Identify common regulons and unique regulons
print("Identifying common and unique regulons across treatments...")

# Get all regulons from each treatment
regulon_sets = {t: set(auc_scores_dict[t].columns) for t in auc_scores_dict}

# Find regulons present in all treatments (common core)
common_regulons = set.intersection(*regulon_sets.values()) if regulon_sets else set()
print(f"Found {len(common_regulons)} regulons common to all treatments")

# Find regulons unique to each treatment
unique_regulons = {}
for treatment, regulon_set in regulon_sets.items():
    other_regulons = set.union(*[rs for t, rs in regulon_sets.items() if t != treatment])
    unique_regulons[treatment] = regulon_set - other_regulons
    print(f"  {treatment}: {len(unique_regulons[treatment])} unique regulons")

# Create a set diagram of regulon overlap
# First, create a binary presence/absence matrix for regulons across treatments
all_regulons = set.union(*regulon_sets.values())
regulon_presence = pd.DataFrame(index=sorted(all_regulons), columns=treatment_groups)

for treatment in treatment_groups:
    if treatment in regulon_sets:
        for regulon in all_regulons:
            regulon_presence.loc[regulon, treatment] = 1 if regulon in regulon_sets[treatment] else 0

# Count how many treatments each regulon appears in
regulon_presence['num_treatments'] = regulon_presence.sum(axis=1)

# Create bar plot of regulon sharing
plt.figure(figsize=(10, 6))
counts = regulon_presence['num_treatments'].value_counts().sort_index()
plt.bar(counts.index, counts.values)
plt.xlabel('Number of Treatment Groups')
plt.ylabel('Number of Regulons')
plt.title('Regulon Sharing Across Treatment Groups')
plt.xticks(range(1, len(treatment_groups) + 1))
plt.tight_layout()
plt.savefig(os.path.join(COMPARISON_DIR, "regulon_sharing_barplot.png"), dpi=300)
plt.close()

# Save lists of common and unique regulons
with open(os.path.join(COMPARISON_DIR, "common_and_unique_regulons.txt"), 'w') as f:
    f.write("COMMON REGULONS (present in all treatments)\n")
    f.write("===========================================\n")
    for reg in sorted(common_regulons):
        f.write(f"{reg}\n")
    
    f.write("\n\nUNIQUE REGULONS (present in only one treatment)\n")
    f.write("============================================\n")
    for treatment in treatment_groups:
        if treatment in unique_regulons:
            f.write(f"\n{treatment}:\n")
            for reg in sorted(unique_regulons[treatment]):
                f.write(f"  {reg}\n")

# %% 3. Compare regulon structure (target genes)
print("Comparing regulon structure across treatments...")

# Select common regulons for comparing target genes
if len(common_regulons) > 0:
    # Create dict to store regulon target genes for each treatment
    regulon_targets = {}
    
    for treatment in treatment_groups:
        if treatment in regulons_dict:
            regulon_targets[treatment] = {}
            for regulon in regulons_dict[treatment]:
                if regulon.name in common_regulons:
                    # Store target genes
                    regulon_targets[treatment][regulon.name] = set(regulon.genes)
    
    # Calculate Jaccard similarity for target gene sets between treatments
    # For each common regulon, calculate similarity matrix across treatments
    regulon_similarity = {}
    
    for regulon_name in common_regulons:
        # Create matrix for this regulon
        treatments_with_regulon = [t for t in treatment_groups if t in regulon_targets and regulon_name in regulon_targets[t]]
        if len(treatments_with_regulon) < 2:
            continue
            
        similarity_matrix = pd.DataFrame(index=treatments_with_regulon, columns=treatments_with_regulon)
        
        # Fill similarity matrix
        for t1 in treatments_with_regulon:
            for t2 in treatments_with_regulon:
                if t1 == t2:
                    similarity_matrix.loc[t1, t2] = 1.0
                else:
                    # Calculate Jaccard similarity
                    genes1 = regulon_targets[t1][regulon_name]
                    genes2 = regulon_targets[t2][regulon_name]
                    
                    if not genes1 or not genes2:
                        similarity_matrix.loc[t1, t2] = 0
                    else:
                        similarity_matrix.loc[t1, t2] = len(genes1.intersection(genes2)) / len(genes1.union(genes2))
        
        regulon_similarity[regulon_name] = similarity_matrix
    
    # Calculate average similarity across all common regulons
    all_treatments = [t for t in treatment_groups if t in regulon_targets]
    avg_similarity = pd.DataFrame(index=all_treatments, columns=all_treatments, dtype=float)
    
    # Initialize with zeros
    for t1 in all_treatments:
        for t2 in all_treatments:
            avg_similarity.loc[t1, t2] = 0.0
    
    # Sum similarities across regulons
    regulon_count = 0
    for regulon_name, sim_matrix in regulon_similarity.items():
        treatments_in_matrix = sim_matrix.index
        
        for t1 in treatments_in_matrix:
            for t2 in treatments_in_matrix:
                avg_similarity.loc[t1, t2] += sim_matrix.loc[t1, t2]
        
        regulon_count += 1
    
    # Calculate average
    if regulon_count > 0:
        avg_similarity = avg_similarity / regulon_count
    
    # Ensure similarity matrix contains numeric values
    avg_similarity_numeric = avg_similarity.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # Plot average similarity heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(avg_similarity_numeric, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1)
    plt.title('Average Regulon Target Gene Similarity Between Treatments')
    for label in ax.get_xticklabels():
        label.set_text(treatment_labels.get(label.get_text(), label.get_text()))
    for label in ax.get_yticklabels():
        label.set_text(treatment_labels.get(label.get_text(), label.get_text()))
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "avg_regulon_target_similarity.png"), dpi=300)
    plt.close()
    
    # Plot individual heatmaps for top 5 most variable regulons
    regulon_variability = {}
    
    for regulon_name, sim_matrix in regulon_similarity.items():
        # Calculate variability as mean of (1 - similarity) for off-diagonal elements
        off_diag_values = []
        for i, row in sim_matrix.iterrows():
            for j, val in row.items():
                if i != j:  # Off-diagonal
                    off_diag_values.append(1 - val)
        
        if off_diag_values:
            regulon_variability[regulon_name] = np.mean(off_diag_values)
    
    # Sort by variability
    sorted_regulons = sorted(regulon_variability.items(), key=lambda x: x[1], reverse=True)
    
    # Plot top 10 most variable regulons in 2 rows of 5
    top_n = min(10, len(sorted_regulons))
    if top_n > 0:
        # Calculate number of rows and columns
        n_cols = 5
        n_rows = 2  # Fixed 2 rows
        
        # Use a larger figure size with more height per row
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*5.5))
        
        # Flatten axes array for easier indexing
        axes = axes.flatten()
        
        for i, (regulon_name, variability) in enumerate(sorted_regulons[:top_n]):
            if i >= len(axes):
                break
                
            ax = axes[i]
            sim_matrix = regulon_similarity[regulon_name]
            
            # Ensure matrix contains numeric values before plotting
            sim_matrix_numeric = sim_matrix.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
            
            # Plot heatmap with adjusted parameters
            sns.heatmap(sim_matrix_numeric, annot=True, fmt=".2f", cmap="YlGnBu", 
                      vmin=0, vmax=1, ax=ax, cbar=False, annot_kws={"size": 9})
            
            # Set title
            clean_name = regulon_name.replace('(+)', '')
            ax.set_title(f"{clean_name}\nVariability: {variability:.2f}", fontsize=12, pad=10)
            
            # Adjust tick labels with improved placement
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            
            # Replace tick labels with treatment labels
            ax.set_xticklabels([treatment_labels.get(t, t) for t in sim_matrix.columns])
            ax.set_yticklabels([treatment_labels.get(t, t) for t in sim_matrix.index])
        
        # Hide any unused subplots if top_n < 10
        for j in range(top_n, n_rows * n_cols):
            if j < len(axes):
                axes[j].axis('off')
        
        # Add more spacing between subplots
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARISON_DIR, "top_variable_regulons_similarity.png"), dpi=300, bbox_inches='tight')
        plt.close()

# %% 4. Compare regulon activity (AUC) across treatments
print("Comparing regulon activity across treatments...")

# Create a combined dataset of average regulon activity per treatment
avg_auc_per_treatment = {}

for treatment in treatment_groups:
    if treatment in auc_scores_dict:
        # Calculate average AUC for each regulon
        avg_auc_per_treatment[treatment] = auc_scores_dict[treatment].mean()

# Convert to DataFrame for easier manipulation
if avg_auc_per_treatment:
    avg_auc_df = pd.DataFrame(avg_auc_per_treatment)
    
    # Fill NAs with zeros (regulons not present in a treatment)
    avg_auc_df = avg_auc_df.fillna(0)
    
    # Save to CSV
    avg_auc_df.to_csv(os.path.join(COMPARISON_DIR, "avg_regulon_activity_by_treatment.csv"))
    
    # Create heatmap of regulon activity across treatments
    # Select top 50 most variable regulons for visualization
    regulon_variance = avg_auc_df.var(axis=1).sort_values(ascending=False)
    top_variable_regulons = regulon_variance.head(50).index
    
    # Create heatmap
    plt.figure(figsize=(10, 16))
    g = sns.clustermap(
        avg_auc_df.loc[top_variable_regulons],
        cmap="YlOrRd",
        figsize=(10, 16),
        xticklabels=True,
        yticklabels=True,
        col_cluster=False  # Don't cluster treatments for better interpretation
    )
    
    # Update x-axis labels to use treatment_labels
    for label in g.ax_heatmap.get_xticklabels():
        label.set_text(treatment_labels.get(label.get_text(), label.get_text()))
        label.set_rotation(45)
        label.set_ha('right')
    
    # Clean up y-axis labels (remove (+))
    ylabels = [label.get_text().replace('(+)', '') for label in g.ax_heatmap.get_yticklabels()]
    g.ax_heatmap.set_yticklabels(ylabels)
    
    plt.suptitle("Top 50 Variable Regulons Across Treatments", y=1.0, fontsize=16)
    plt.savefig(os.path.join(COMPARISON_DIR, "top_variable_regulon_activity_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar plots for top 10 most variable regulons
    top_10_regulons = regulon_variance.head(10).index
    
    # Set up the figure
    fig, axes = plt.subplots(5, 2, figsize=(14, 16))
    axes = axes.flatten()
    
    for i, regulon in enumerate(top_10_regulons):
        ax = axes[i]
        
        # Get data for this regulon
        data = avg_auc_df.loc[regulon]
        
        # Bar plot
        bars = ax.bar(data.index, data.values, color=[treatment_colors.get(t, 'gray') for t in data.index])
        
        # Set title and labels
        clean_name = regulon.replace('(+)', '')
        ax.set_title(clean_name, fontsize=12)
        ax.set_ylabel('Average AUC')
        
        # Replace x-tick labels with treatment_labels
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels([treatment_labels.get(t, t) for t in data.index], rotation=45, ha='right')
    
    # Add a main title
    plt.suptitle("Average Activity of Top 10 Most Variable Regulons", y=0.98, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "top10_variable_regulons_barplots.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create PCA plot of treatments based on regulon activity
    if len(avg_auc_df.columns) >= 3:  # Need at least 3 treatments for PCA to be meaningful
        # Standardize the data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        regulon_activity_scaled = scaler.fit_transform(avg_auc_df.T)
        
        # Run PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(regulon_activity_scaled)
        
        # Create a DataFrame with PCA results
        pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'], index=avg_auc_df.columns)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        for treatment in pca_df.index:
            plt.scatter(
                pca_df.loc[treatment, 'PC1'],
                pca_df.loc[treatment, 'PC2'],
                s=100,
                color=treatment_colors.get(treatment, 'gray'),
                label=treatment_labels.get(treatment, treatment)
            )
            
            # Add treatment label
            plt.annotate(
                treatment_labels.get(treatment, treatment),
                xy=(pca_df.loc[treatment, 'PC1'], pca_df.loc[treatment, 'PC2']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=12
            )
        
        # Add axis labels with variance explained
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        
        plt.title('PCA of Treatments Based on Regulon Activity', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARISON_DIR, "treatment_pca_regulon_activity.png"), dpi=300)
        plt.close()

# %% 4.1 Plot all regulon AUC values scaled to percentage-like representation
print("Creating scaled heatmap of all regulon activity across treatments...")

# Create a directory for scaled AUC visualizations
scaled_auc_dir = os.path.join(COMPARISON_DIR, "scaled_auc")
os.makedirs(scaled_auc_dir, exist_ok=True)

# Use the existing avg_auc_df which contains average AUC scores per treatment
if avg_auc_per_treatment and not avg_auc_df.empty:
    # Create a scaled version by multiplying by 100
    scaled_auc_df = avg_auc_df * 100
    
    # Save scaled version to CSV
    scaled_auc_df.to_csv(os.path.join(scaled_auc_dir, "scaled_regulon_activity_by_treatment.csv"))
    
    # Create a heatmap with all regulons, not just variable ones
    # But sort by variance for better visualization
    regulon_variance = avg_auc_df.var(axis=1).sort_values(ascending=False)
    sorted_regulons = regulon_variance.index
    
    # Create full heatmap - might be very tall, so adjust figure size accordingly
    plt.figure(figsize=(10, max(16, len(sorted_regulons)*0.16)))
    g = sns.clustermap(
        scaled_auc_df.loc[sorted_regulons],
        cmap="flare",
        figsize=(10, max(16, len(sorted_regulons)*0.16)),
        xticklabels=True,
        yticklabels=True,
        col_cluster=False,  # Don't cluster treatments for better interpretation
        row_cluster=True   # Keep the variance-based ordering
    )
    
    # Update x-axis labels to use treatment_labels
    for label in g.ax_heatmap.get_xticklabels():
        label.set_text(treatment_labels.get(label.get_text(), label.get_text()))
        label.set_rotation(45)
        label.set_ha('right')
    
    # Clean up y-axis labels (remove (+))
    ylabels = [label.get_text().replace('(+)', '') for label in g.ax_heatmap.get_yticklabels()]
    g.ax_heatmap.set_yticklabels(ylabels)
    
    plt.suptitle("All Regulons Across Treatments (AUC × 100)", y=1.0, fontsize=16)
    plt.savefig(os.path.join(scaled_auc_dir, "all_regulon_activity_heatmap_scaled.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # If there are too many regulons, also create a version with top 100 for easier viewing
    if len(sorted_regulons) > 100:
        top_100_regulons = sorted_regulons[:100]
        
        plt.figure(figsize=(10, 24))
        g = sns.clustermap(
            scaled_auc_df.loc[top_100_regulons],
            cmap="YlOrRd",
            figsize=(10, 24),
            xticklabels=True,
            yticklabels=True,
            col_cluster=False,  # Don't cluster treatments for better interpretation
            row_cluster=True   # Keep the variance-based ordering
        )
        
        # Update x-axis labels to use treatment_labels
        for label in g.ax_heatmap.get_xticklabels():
            label.set_text(treatment_labels.get(label.get_text(), label.get_text()))
            label.set_rotation(45)
            label.set_ha('right')
        
        # Clean up y-axis labels (remove (+))
        ylabels = [label.get_text().replace('(+)', '') for label in g.ax_heatmap.get_yticklabels()]
        g.ax_heatmap.set_yticklabels(ylabels)
        
        plt.suptitle("Top 100 Variable Regulons Across Treatments (AUC × 100)", y=1.0, fontsize=16)
        plt.savefig(os.path.join(scaled_auc_dir, "top100_regulon_activity_heatmap_scaled.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a summary table with percentile ranks that preserves zeros
    print("Creating percentile-ranked regulon activity table...")
    
    # Calculate percentile ranks within each treatment, but preserve zeros
    percentile_ranks = pd.DataFrame(index=scaled_auc_df.index)
    for treatment in scaled_auc_df.columns:
        # First identify zero values
        is_zero = scaled_auc_df[treatment] == 0
        
        # Rank only non-zero values
        temp_ranks = scaled_auc_df[treatment].copy()
        temp_ranks[is_zero] = np.nan  # Temporarily mark zeros as NaN
        ranked = temp_ranks.rank(pct=True) * 100
        
        # Put zeros back as zeros
        ranked[is_zero] = 0
        
        # Add to dataframe
        percentile_ranks[treatment] = ranked
    
    # Save percentile ranks
    percentile_ranks.to_csv(os.path.join(scaled_auc_dir, "regulon_percentile_ranks.csv"))
    
    # Plot heatmap of percentile ranks for top 100 variable regulons
    plt.figure(figsize=(10, 24))
    g = sns.clustermap(
        percentile_ranks.loc[top_100_regulons] if len(sorted_regulons) > 100 else percentile_ranks,
        cmap="flare",
        figsize=(10, 24),
        xticklabels=True,
        yticklabels=True,
        col_cluster=False,
        row_cluster=True
    )
    
    # Update x-axis labels to use treatment_labels
    for label in g.ax_heatmap.get_xticklabels():
        label.set_text(treatment_labels.get(label.get_text(), label.get_text()))
        label.set_rotation(45)
        label.set_ha('right')
    
    # Clean up y-axis labels (remove (+))
    ylabels = [label.get_text().replace('(+)', '') for label in g.ax_heatmap.get_yticklabels()]
    g.ax_heatmap.set_yticklabels(ylabels)
    
    plt.suptitle("Regulon Activity Percentile Ranks Within Each Treatment\n(Zero values preserved as zero)", y=1.0, fontsize=16)
    plt.savefig(os.path.join(scaled_auc_dir, "regulon_percentile_ranks_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Scaled regulon activity visualization complete!")
else:
    print("No average AUC data available for scaling")

# %% 5. Compare cell type-specific regulon activity
print("Comparing cell type-specific regulon activity...")

# First, identify common cell types across treatments
cell_types_per_treatment = {}

for treatment in treatment_groups:
    if treatment in celltype_avg_auc_dict:
        cell_types_per_treatment[treatment] = set(celltype_avg_auc_dict[treatment].index)

# Find common cell types
common_cell_types = set.intersection(*cell_types_per_treatment.values()) if cell_types_per_treatment else set()
print(f"Found {len(common_cell_types)} cell types common to all treatments")

# For each common cell type, compare regulon activity across treatments
if common_cell_types and common_regulons:
    os.makedirs(os.path.join(COMPARISON_DIR, "cell_type_comparisons"), exist_ok=True)
    
    for cell_type in sorted(common_cell_types):
        print(f"Comparing regulon activity for cell type: {cell_type}")
        
        # Create a DataFrame with regulon activity for this cell type across treatments
        cell_type_activity = pd.DataFrame(index=sorted(common_regulons))
        
        for treatment in treatment_groups:
            if treatment in celltype_avg_auc_dict:
                # Extract data for this cell type
                cell_data = celltype_avg_auc_dict[treatment].loc[cell_type]
                
                # Add common regulons
                for regulon in common_regulons:
                    if regulon in cell_data:
                        cell_type_activity.loc[regulon, treatment] = cell_data[regulon]
                    else:
                        cell_type_activity.loc[regulon, treatment] = 0
        
        # Fill NAs with zeros
        cell_type_activity = cell_type_activity.fillna(0)
        
        # Save to CSV
        cell_type_activity.to_csv(os.path.join(COMPARISON_DIR, "cell_type_comparisons", f"{make_valid_filename(cell_type)}_activity.csv"))
        
        # Create heatmap
        plt.figure(figsize=(10, 12))
        g = sns.clustermap(
            cell_type_activity,
            cmap="YlOrRd",
            figsize=(10, 12),
            xticklabels=True,
            yticklabels=True,
            col_cluster=False,  # Don't cluster treatments
            row_cluster=True    # Cluster regulons
        )
        
        # Update x-axis labels
        for label in g.ax_heatmap.get_xticklabels():
            label.set_text(treatment_labels.get(label.get_text(), label.get_text()))
            label.set_rotation(45)
            label.set_ha('right')
        
        # Clean up y-axis labels (remove (+))
        ylabels = [label.get_text().replace('(+)', '') for label in g.ax_heatmap.get_yticklabels()]
        g.ax_heatmap.set_yticklabels(ylabels)
        
        plt.suptitle(f"Regulon Activity in {cell_type} Cells Across Treatments", y=1.0, fontsize=16)
        plt.savefig(os.path.join(COMPARISON_DIR, "cell_type_comparisons", f"{make_valid_filename(cell_type)}_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # For this cell type, identify differentially active regulons
        # For each regulon, perform ANOVA across treatments if we have enough data
        if len(cell_type_activity.columns) >= 3:  # Need at least 3 treatments for ANOVA
            # Identify regulons with highest variance across treatments
            regulon_variance = cell_type_activity.var(axis=1).sort_values(ascending=False)
            top_variable_regulons = regulon_variance.head(10).index
            
            # Create bar plots for top 10 variable regulons
            fig, axes = plt.subplots(5, 2, figsize=(14, 16))
            axes = axes.flatten()
            
            for i, regulon in enumerate(top_variable_regulons):
                ax = axes[i]
                
                # Get data for this regulon
                data = cell_type_activity.loc[regulon]
                
                # Bar plot
                bars = ax.bar(data.index, data.values, color=[treatment_colors.get(t, 'gray') for t in data.index])
                
                # Set title and labels
                clean_name = regulon.replace('(+)', '')
                ax.set_title(f"{clean_name}", fontsize=12)
                ax.set_ylabel('Activity')
                
                # Replace x-tick labels
                ax.set_xticks(range(len(data.index)))
                ax.set_xticklabels([treatment_labels.get(t, t) for t in data.index], rotation=45, ha='right')
            
            # Add a main title
            plt.suptitle(f"Top Variable Regulons in {cell_type} Cells", y=0.98, fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(COMPARISON_DIR, "cell_type_comparisons", f"{make_valid_filename(cell_type)}_top10_barplots.png"), dpi=300, bbox_inches='tight')
            plt.close()

# %% 6. Generate comparison network visualization
print("Generating comparative network visualization...")

# Create a combined network of regulons across all treatments
if regulons_dict:
    # First, collect all TF to target gene connections
    tf_target_connections = defaultdict(lambda: defaultdict(int))
    
    for treatment, regulons in regulons_dict.items():
        for regulon in regulons:
            # Extract TF name without (+)
            tf_name = regulon.name.replace('(+)', '')
            
            # Count each target gene connection
            for target_gene in regulon.genes:
                tf_target_connections[tf_name][target_gene] += 1
    
    # Create network with edges weighted by frequency across treatments
    G = nx.DiGraph()
    
    # Add edges with weight based on how many treatments have this connection
    for tf, targets in tf_target_connections.items():
        for target, count in targets.items():
            # Only include edges that appear in multiple treatments
            if count > 1:
                G.add_edge(tf, target, weight=count, penwidth=0.5 + count * 0.5)
    
    # Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    
    # Limit to manageable size for visualization
    if len(G.nodes()) > 200:
        # Keep only the nodes with highest degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:200]
        top_node_names = [node for node, degree in top_nodes]
        
        # Create subgraph with only top nodes
        G = G.subgraph(top_node_names)
    
    print(f"Created network with {len(G.nodes())} nodes and {len(G.edges())} edges")
    
    # Compute node properties
    node_degrees = dict(G.degree())
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    
    # Calculate node sizes based on degree
    node_sizes = {node: 100 + 10 * degree for node, degree in node_degrees.items()}
    
    # Identify TFs (nodes with outgoing edges)
    tfs = [node for node, out_deg in out_degrees.items() if out_deg > 0]
    
    # Generate position layout (can be slow for large networks)
    if len(G.nodes()) > 100:
        # For larger networks, use a faster layout algorithm
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    else:
        # For smaller networks, use a more accurate layout
        pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Create visualization
    plt.figure(figsize=(16, 16))
    
    # Draw edges with varying thickness
    for u, v, data in G.edges(data=True):
        # Get edge weight
        weight = data.get('weight', 1)
        # Calculate edge width and alpha based on weight
        width = 0.5 + 0.5 * weight
        alpha = 0.3 + 0.1 * weight
        
        # Draw edge
        plt.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], 
                color='gray', alpha=alpha, linewidth=width, zorder=1)
    
    # Draw nodes
    # Differentiate between TFs and target genes
    tf_nodes = [node for node in G.nodes() if node in tfs]
    target_nodes = [node for node in G.nodes() if node not in tfs]
    
    # Draw TF nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=tf_nodes,
                          node_size=[node_sizes.get(node, 100) for node in tf_nodes],
                          node_color='#ff7f0e',  # Orange for TFs
                          alpha=0.8,
                          label='Transcription Factors')
    
    # Draw target nodes
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=target_nodes,
                          node_size=[node_sizes.get(node, 50) for node in target_nodes],
                          node_color='#1f77b4',  # Blue for targets
                          alpha=0.6,
                          label='Target Genes')
    
    # Add labels only for TFs and high-degree target genes
    important_nodes = [node for node in G.nodes() if node in tfs or node_degrees.get(node, 0) > 5]
    if len(important_nodes) > 50:
        important_nodes = important_nodes[:50]  # Limit labels to avoid cluttering
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                          {node: node for node in important_nodes},
                          font_size=10,
                          font_weight='bold')
    
    plt.title("Cross-Treatment Regulon Network", fontsize=16)
    plt.legend()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "cross_treatment_regulon_network.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a subnetwork focusing on specific TFs (optional)
    # Get top 5 TFs by out-degree
    top_tfs = sorted([(node, degree) for node, degree in out_degrees.items()], 
                   key=lambda x: x[1], reverse=True)[:5]
    
    if top_tfs:
        # Create subgraph
        top_tf_names = [tf for tf, _ in top_tfs]
        
        for tf in top_tf_names:
            # Get all neighbors of this TF
            neighbors = list(G.successors(tf))
            
            # Create subgraph
            sub_nodes = [tf] + neighbors
            sub_graph = G.subgraph(sub_nodes)
            
            # Generate layout
            sub_pos = nx.spring_layout(sub_graph, k=0.5, seed=42)
            
            plt.figure(figsize=(12, 10))
            
            # Draw edges
            for u, v, data in sub_graph.edges(data=True):
                weight = data.get('weight', 1)
                width = 0.5 + 0.5 * weight
                alpha = 0.3 + 0.1 * weight
                
                plt.plot([sub_pos[u][0], sub_pos[v][0]], [sub_pos[u][1], sub_pos[v][1]], 
                        color='gray', alpha=alpha, linewidth=width, zorder=1)
            
            # Draw nodes
            nx.draw_networkx_nodes(sub_graph, sub_pos, 
                                  nodelist=[tf],
                                  node_size=500,
                                  node_color='#ff7f0e',
                                  alpha=0.8)
            
            nx.draw_networkx_nodes(sub_graph, sub_pos, 
                                  nodelist=neighbors,
                                  node_size=200,
                                  node_color='#1f77b4',
                                  alpha=0.6)
            
            # Draw labels
            nx.draw_networkx_labels(sub_graph, sub_pos, 
                                  font_size=10,
                                  font_weight='bold')
            
            plt.title(f"{tf} Regulon Network", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(COMPARISON_DIR, f"{make_valid_filename(tf)}_subnetwork.png"), dpi=300, bbox_inches='tight')
            plt.close()

# %% 7. Treatment-specific enrichment of regulons
print("Analyzing treatment-specific regulon enrichment...")

# For each treatment, identify the most active regulons compared to other treatments
treatment_enriched_regulons = {}

if avg_auc_df is not None and not avg_auc_df.empty:
    # For each regulon, calculate fold change between each treatment and the average of others
    for treatment in treatment_groups:
        if treatment not in avg_auc_df.columns:
            continue
            
        treatment_enriched_regulons[treatment] = []
        
        for regulon in avg_auc_df.index:
            # Get activity in this treatment
            activity_treatment = avg_auc_df.loc[regulon, treatment]
            
            # Get average activity in other treatments
            other_treatments = [t for t in avg_auc_df.columns if t != treatment]
            if other_treatments:
                activity_others = avg_auc_df.loc[regulon, other_treatments].mean()
                
                # Calculate fold change (avoid division by zero)
                if activity_others > 0:
                    fold_change = activity_treatment / activity_others
                else:
                    if activity_treatment > 0:
                        fold_change = float('inf')  # Infinity for cases where only this treatment has activity
                    else:
                        fold_change = 1.0  # Both are zero
                
                # Store result if fold change is significant
                if fold_change > 1.5:  # Arbitrary threshold, can be adjusted
                    treatment_enriched_regulons[treatment].append((regulon, fold_change))
        
        # Sort by fold change
        treatment_enriched_regulons[treatment].sort(key=lambda x: x[1], reverse=True)
        
        print(f"  {treatment}: {len(treatment_enriched_regulons[treatment])} enriched regulons")
    
    # Create enrichment visualization
    # Bar plot of top 10 enriched regulons for each treatment
    for treatment, enriched_regs in treatment_enriched_regulons.items():
        if not enriched_regs:
            continue
            
        # Take top 10 or all if less than 10
        top_regs = enriched_regs[:min(10, len(enriched_regs))]
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Regulon': [reg[0].replace('(+)', '') for reg in top_regs],
            'Fold Change': [reg[1] for reg in top_regs]
        })
        
        # Sort by fold change
        plot_data = plot_data.sort_values('Fold Change', ascending=True)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(plot_data['Regulon'], plot_data['Fold Change'], 
                       color=treatment_colors.get(treatment, 'blue'))
        
        # Add fold change values at the end of each bar
        for i, v in enumerate(plot_data['Fold Change']):
            if v > 10:  # For very large values
                plt.text(10, i, f"FC: {v:.1f}", va='center')
            else:
                plt.text(v + 0.1, i, f"{v:.1f}", va='center')
        
        # Set reasonable x-axis limit
        plt.xlim(0, min(max(plot_data['Fold Change']) * 1.2, 10))
        
        plt.title(f"Top Enriched Regulons in {treatment_labels.get(treatment, treatment)}", fontsize=14)
        plt.xlabel('Fold Change vs Other Treatments')
        plt.tight_layout()
        plt.savefig(os.path.join(COMPARISON_DIR, f"{make_valid_filename(treatment)}_enriched_regulons.png"), dpi=300)
        plt.close()
    
    # Save enriched regulons to file
    with open(os.path.join(COMPARISON_DIR, "treatment_enriched_regulons.txt"), 'w') as f:
        f.write("TREATMENT-ENRICHED REGULONS\n")
        f.write("===========================\n\n")
        
        for treatment, enriched_regs in treatment_enriched_regulons.items():
            f.write(f"{treatment} ({treatment_labels.get(treatment, treatment)}):\n")
            f.write("-" * 50 + "\n")
            
            for i, (regulon, fold_change) in enumerate(enriched_regs[:20]):  # Show top 20
                f.write(f"{i+1}. {regulon} (Fold Change: {fold_change:.2f})\n")
            
            f.write("\n\n")

# %% 8. Pairwise treatment comparisons
print("Performing pairwise treatment comparisons...")

# Create a directory for pairwise comparisons
pairwise_dir = os.path.join(COMPARISON_DIR, "pairwise_comparisons")
os.makedirs(pairwise_dir, exist_ok=True)

# For each pair of treatments, compare regulon activity
if avg_auc_df is not None and len(avg_auc_df.columns) >= 2:
    treatments = list(avg_auc_df.columns)
    
    for i, t1 in enumerate(treatments):
        for j, t2 in enumerate(treatments):
            if i < j:  # Only compare each pair once
                print(f"Comparing {t1} vs {t2}...")
                
                # Get regulons present in both treatments - convert set to list explicitly
                if t1 in regulon_sets and t2 in regulon_sets:
                    common_regulons_list = list(regulon_sets[t1].intersection(regulon_sets[t2]))
                    
                    if common_regulons_list:  # Only proceed if there are common regulons
                        # Create scatter plot of regulon activity
                        plt.figure(figsize=(12, 12))  # Increased figure size
                        
                        # Extract activities - using list indexing
                        x = avg_auc_df.loc[common_regulons_list, t1]
                        y = avg_auc_df.loc[common_regulons_list, t2]
                        
                        # Plot scatter with larger points and improved styling
                        plt.scatter(x, y, alpha=0.7, s=80, color='#1f77b4', edgecolor='white', linewidth=0.5)
                        
                        # Add diagonal line
                        max_val = max(x.max(), y.max()) * 1.1
                        plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1.5)
                        
                        # Add labels for top differentially active regulons with improved visibility
                        diff = abs(x - y)
                        top_diff_idx = diff.nlargest(10).index
                        
                        for regulon in top_diff_idx:
                            # Highlight the points we're labeling
                            plt.scatter(x[regulon], y[regulon], s=150, color='#FF6DB9', 
                                       edgecolor='black', linewidth=1, zorder=10)
                                       
                            # Add text label with better formatting and positioning
                            plt.annotate(
                                regulon.replace('(+)', ''),
                                xy=(x[regulon], y[regulon]),
                                xytext=(10, 10),  # Increased offset
                                textcoords='offset points',
                                fontsize=24,  # Larger font size
                                fontweight='bold',
                                color='#000000',
                                bbox=dict(boxstyle="round,pad=0.3", fc='#ffffffaa', ec="none", alpha=0.7)
                            )
                        
                        # Set labels and title with improved styling
                        plt.xlabel(f'{treatment_labels.get(t1, t1)} Regulon Activity', fontsize=14)
                        plt.ylabel(f'{treatment_labels.get(t2, t2)} Regulon Activity', fontsize=14)
                        plt.title(f'Regulon Activity: {treatment_labels.get(t1, t1)} vs {treatment_labels.get(t2, t2)}', 
                                 fontsize=16, fontweight='bold')
                        
                        # Add correlation coefficient with improved formatting
                        corr = x.corr(y)
                        plt.annotate(
                            f'r = {corr:.3f}',
                            xy=(0.05, 0.95),
                            xycoords='axes fraction',
                            fontsize=14,
                            fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="lightgray")
                        )
                        
                        # Add gridlines for better readability
                        plt.grid(True, linestyle='--', alpha=0.3)
                        
                        # Ensure axes start at zero and have matching scales for fair comparison
                        plt.xlim(0, max_val)
                        plt.ylim(0, max_val)
                        
                        # Format tick marks
                        plt.tick_params(axis='both', which='major', labelsize=12)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(pairwise_dir, f"{make_valid_filename(t1)}_vs_{make_valid_filename(t2)}_scatter.png"), 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # Create a table of top differential regulons
                        diff_df = pd.DataFrame({
                            t1: x,
                            t2: y,
                            'Difference': x - y,
                            'Abs Difference': abs(x - y),
                            'Fold Change': y / x.replace(0, float('nan'))
                        })
                        
                        # Sort by absolute difference
                        diff_df = diff_df.sort_values('Abs Difference', ascending=False)
                        
                        # Save top differential regulons
                        diff_df.head(50).to_csv(os.path.join(pairwise_dir, f"{make_valid_filename(t1)}_vs_{make_valid_filename(t2)}_diff_regulons.csv"))
                    else:
                        print(f"  No common regulons found between {t1} and {t2}")
                else:
                    print(f"  Missing regulon data for either {t1} or {t2}")

# %% 9. Cluster treatments based on regulon activity profiles
print("Clustering treatments based on regulon activity profiles...")

# Use hierarchical clustering to group treatments based on regulon activity
if avg_auc_df is not None and len(avg_auc_df.columns) >= 3:
    # Transpose so treatments are rows
    treatment_profiles = avg_auc_df.T
    
    # Hierarchical clustering
    plt.figure(figsize=(12, 8))
    dendrogram = hierarchy.dendrogram(
        hierarchy.linkage(treatment_profiles, method='average'),
        labels=[treatment_labels.get(t, t) for t in treatment_profiles.index],
        leaf_rotation=90
    )
    
    plt.title('Hierarchical Clustering of Treatments Based on Regulon Activity', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "treatment_clustering_dendrogram.png"), dpi=300)
    plt.close()
    
    # Create a similarity matrix between treatments
    similarity_matrix = pd.DataFrame(index=treatment_profiles.index, columns=treatment_profiles.index)
    
    for t1 in treatment_profiles.index:
        for t2 in treatment_profiles.index:
            # Calculate Pearson correlation
            corr = treatment_profiles.loc[t1].corr(treatment_profiles.loc[t2])
            similarity_matrix.loc[t1, t2] = corr
    
    # Convert similarity matrix to numeric type before plotting
    similarity_matrix_numeric = similarity_matrix.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    
    # Plot similarity heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        similarity_matrix_numeric,  # Use numeric version
        cmap="YlOrRd",
        vmin=0.5,  # Adjust vmin to highlight differences
        vmax=1.0,
        annot=True,
        fmt=".2f"
    )
    
    # Update tick labels
    ax.set_xticklabels([treatment_labels.get(t, t) for t in similarity_matrix.columns], rotation=45, ha='right')
    ax.set_yticklabels([treatment_labels.get(t, t) for t in similarity_matrix.index], rotation=0)
    
    plt.title('Treatment Similarity Based on Regulon Activity', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, "treatment_similarity_heatmap.png"), dpi=300)
    plt.close()

# %% 10. Transcription factor-specific analysis
print("Performing enhanced analysis of specific transcription factors...")

# Define TFs of interest - MODIFY THIS SECTION FOR YOUR ANALYSIS
tfs_of_interest = ["Ezh2(+)"
    # DEFINE YOUR TFs OF INTEREST HERE
    # Example: ["Ezh2(+)", "Tcf7(+)", "Foxp3(+)"]
]

if tfs_of_interest:
    # Function to find matching TF names in regulons
    def find_matching_tfs(tf_names, regulon_names):
        """Find all regulon names containing any of the specified TF names"""
        matched_regulons = []
        for regulon in regulon_names:
            # Clean regulon name for matching
            clean_reg = regulon.replace("(+)", "").lower()
            # Check if it matches any of our TFs of interest
            for tf in tf_names:
                clean_tf = tf.replace("(+)", "").lower()
                if clean_tf == clean_reg:
                    matched_regulons.append(regulon)
                    break
        return matched_regulons

    # Create a directory for TF-specific analyses
    tf_analysis_dir = os.path.join(COMPARISON_DIR, "tf_specific_analysis")
    os.makedirs(tf_analysis_dir, exist_ok=True)

    # For each TF of interest, find matching regulons across treatments
    matched_regulons_by_treatment = {}
    for treatment, regulon_set in regulon_sets.items():
        matched = find_matching_tfs(tfs_of_interest, regulon_set)
        if matched:
            matched_regulons_by_treatment[treatment] = matched

    # Aggregate all matched regulons for convenience
    all_matched_regulons = set()
    for treatment, matches in matched_regulons_by_treatment.items():
        all_matched_regulons.update(matches)

    # Check if we found any matches
    if not all_matched_regulons:
        print(f"No regulons found matching the specified TFs: {tfs_of_interest}")
    else:
        print(f"Found {len(all_matched_regulons)} regulons matching the specified TFs across treatments")
        for regulon in sorted(all_matched_regulons):
            print(f"  - {regulon}")
        
        # 1. TF Activity Analysis Across Treatments
        print("\nAnalyzing TF activity across treatments...")
        
        # Create a DataFrame to store average activity for these regulons
        activity_df = pd.DataFrame(index=sorted(all_matched_regulons), columns=treatment_groups)
        
        # Fill in average activity values from the avg_auc_df
        for regulon in activity_df.index:
            for treatment in treatment_groups:
                if treatment in avg_auc_df.columns and regulon in avg_auc_df.index:
                    activity_df.loc[regulon, treatment] = avg_auc_df.loc[regulon, treatment]
        
        # Convert to numeric and fill NaN values with zeros
        activity_df = activity_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Save to CSV
        activity_df.to_csv(os.path.join(tf_analysis_dir, f"{'-'.join(tfs_of_interest).replace('(+)', '')}_activity_across_treatments.csv"))
        
        # Create improved bar plots for TF activity across treatments
        for regulon in activity_df.index:
            plt.figure(figsize=(12, 6))
            
            # Get data for this regulon
            values = activity_df.loc[regulon].values
            
            # Create bar plot with enhanced styling
            bars = plt.bar(
                range(len(treatment_groups)),
                values,
                color=[treatment_colors.get(t, 'gray') for t in treatment_groups],
                width=0.7
            )
            
            # Add value labels on top of bars
            for i, v in enumerate(values):
                if v > 0.01:  # Only label bars with non-negligible values
                    plt.text(i, v + max(values) * 0.02, f"{v:.3f}", 
                             ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Set x-axis ticks and labels
            plt.xticks(range(len(treatment_groups)), 
                       [treatment_labels.get(t, t) for t in treatment_groups], 
                       rotation=45, ha='right', fontsize=11)
            
            # Set title and labels
            clean_name = regulon.replace('(+)', '')
            plt.title(f"{clean_name} Activity Across Treatment Groups", fontsize=16, fontweight='bold')
            plt.ylabel("Average AUC Score", fontsize=14)
            
            # Add grid for better readability
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Adjust y-axis to start from 0 and have some headroom
            max_val = max(values) if max(values) > 0 else 1
            plt.ylim(0, max_val * 1.15)
            
            plt.tight_layout()
            plt.savefig(os.path.join(tf_analysis_dir, f"{make_valid_filename(clean_name)}_activity_across_treatments.png"), dpi=300)
            plt.close()
        
        # 2. Cell Type-Specific TF Activity Analysis
        if celltype_avg_auc_dict:
            print("\nCreating cell type-specific TF activity analysis...")
            
            # Create a directory for cell type analysis
            cell_type_dir = os.path.join(tf_analysis_dir, "cell_type_activity")
            os.makedirs(cell_type_dir, exist_ok=True)
            
            # Collect all cell types across treatments
            all_cell_types = set()
            for treatment, celltype_auc in celltype_avg_auc_dict.items():
                all_cell_types.update(celltype_auc.index)
            
            if all_cell_types:
                print(f"Found {len(all_cell_types)} unique cell types across treatments")
                
                # For each matched regulon, create analysis across cell types and treatments
                for regulon in sorted(all_matched_regulons):
                    clean_name = regulon.replace('(+)', '')
                    print(f"  Creating cell type analysis for {clean_name}...")
                    
                    # Create DataFrame to store cell type-specific activity
                    celltype_activity = pd.DataFrame(index=sorted(all_cell_types), columns=treatment_groups)
                    
                    # Fill in values where available
                    for treatment in treatment_groups:
                        if treatment in celltype_avg_auc_dict:
                            celltype_data = celltype_avg_auc_dict[treatment]
                            
                            # Check if regulon exists in this treatment's data
                            if regulon in celltype_data.columns:
                                for cell_type in all_cell_types:
                                    if cell_type in celltype_data.index:
                                        celltype_activity.loc[cell_type, treatment] = celltype_data.loc[cell_type, regulon]
                    
                    # Fill NaN values with zeros
                    celltype_activity = celltype_activity.apply(pd.to_numeric, errors='coerce').fillna(0)
                    
                    # Save to CSV
                    celltype_activity.to_csv(os.path.join(cell_type_dir, f"{make_valid_filename(clean_name)}_celltype_activity.csv"))
                    
                    # Sort cell types by average activity for better visualization
                    celltype_means = celltype_activity.mean(axis=1)
                    celltype_order = celltype_means.sort_values(ascending=False).index
                    
                    # Create heatmap with cell types sorted by activity
                    plt.figure(figsize=(12, max(8, len(celltype_order) * 0.3)))
                    
                    # Plot heatmap
                    ax = sns.heatmap(
                        celltype_activity.loc[celltype_order], 
                        cmap="YlOrRd",
                        annot=True,
                        fmt=".2f",
                        linewidths=0.5,
                        cbar_kws={"label": "Activity (AUC Score)"}
                    )
                    
                    # Set titles and labels
                    plt.title(f"{clean_name} Activity Across Cell Types and Treatments", fontsize=16)
                    
                    # Improve x-axis labels with treatment labels
                    plt.xticks(np.arange(len(treatment_groups)) + 0.5, 
                              [treatment_labels.get(t, t) for t in treatment_groups], 
                              rotation=45, ha='right')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(cell_type_dir, f"{make_valid_filename(clean_name)}_celltype_heatmap.png"), dpi=300, bbox_inches='tight')
                    plt.close()
            else:
                print("No cell type data available")
        else:
            print("No cell type-specific AUC data found")
        
        # 3. Target Gene Analysis Across Treatments
        if regulons_dict:
            print("\nAnalyzing target genes of the specified TFs across treatments...")
            
            # Collect target genes for each TF regulon across treatments
            target_genes_by_regulon_treatment = {}
            
            for regulon_name in all_matched_regulons:
                target_genes_by_regulon_treatment[regulon_name] = {}
                
                # Find this regulon in each treatment
                for treatment, regulons in regulons_dict.items():
                    for regulon in regulons:
                        if regulon.name == regulon_name:
                            target_genes_by_regulon_treatment[regulon_name][treatment] = set(regulon.genes)
            
            # Analyze target gene overlap between treatments
            target_overlap_dir = os.path.join(tf_analysis_dir, "target_gene_analysis")
            os.makedirs(target_overlap_dir, exist_ok=True)
            
            for regulon_name, treatment_targets in target_genes_by_regulon_treatment.items():
                if len(treatment_targets) < 2:
                    continue  # Skip if only present in one treatment
                
                clean_name = regulon_name.replace('(+)', '')
                print(f"  Analyzing target overlap for {clean_name}...")
                
                # Get list of treatments where this regulon exists
                treatments_with_regulon = list(treatment_targets.keys())
                
                # Create target overlap matrix
                overlap_matrix = pd.DataFrame(index=treatments_with_regulon, columns=treatments_with_regulon, dtype=float)
                
                # Calculate target overlaps (Jaccard index)
                for t1 in treatments_with_regulon:
                    for t2 in treatments_with_regulon:
                        genes1 = treatment_targets[t1]
                        genes2 = treatment_targets[t2]
                        
                        if t1 == t2:
                            overlap_matrix.loc[t1, t2] = 1.0
                        else:
                            # Calculate Jaccard similarity
                            if not genes1 or not genes2:
                                overlap_matrix.loc[t1, t2] = 0.0
                            else:
                                intersection = len(genes1.intersection(genes2))
                                union = len(genes1.union(genes2))
                                overlap_matrix.loc[t1, t2] = float(intersection) / float(union) if union > 0 else 0.0
                
                # Create heatmap of target gene overlap
                plt.figure(figsize=(10, 8))
                ax = sns.heatmap(
                    overlap_matrix.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float), 
                    annot=True, 
                    fmt=".2f", 
                    cmap="YlGnBu", 
                    vmin=0, 
                    vmax=1,
                    linewidths=0.5
                )
                
                # Update axis labels
                ax.set_xticklabels([treatment_labels.get(t, t) for t in overlap_matrix.columns], rotation=45, ha='right')
                ax.set_yticklabels([treatment_labels.get(t, t) for t in overlap_matrix.index], rotation=0)
                
                plt.title(f"{clean_name} Target Gene Overlap Between Treatments\n(Jaccard Similarity Index)", fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(target_overlap_dir, f"{make_valid_filename(clean_name)}_target_overlap.png"), dpi=300)
                plt.close()
                
                # Save target gene breakdown
                target_breakdown_file = os.path.join(target_overlap_dir, f"{make_valid_filename(clean_name)}_target_breakdown.txt")
                with open(target_breakdown_file, 'w') as f:
                    f.write(f"Target Gene Analysis for {regulon_name}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    # Find common targets across all treatments
                    common_targets = set.intersection(*treatment_targets.values()) if treatment_targets else set()
                    
                    f.write("Common Target Genes (present in all treatments):\n")
                    f.write("-" * 50 + "\n")
                    if common_targets:
                        f.write(", ".join(sorted(common_targets)) + "\n")
                        f.write(f"Total: {len(common_targets)} genes\n")
                    else:
                        f.write("No common target genes found across all treatments.\n")
                    
                    f.write("\n")
                    
                    # Write target counts by treatment
                    f.write("Total Target Gene Counts by Treatment:\n")
                    f.write("-" * 50 + "\n")
                    
                    for treatment in treatments_with_regulon:
                        label = treatment_labels.get(treatment, treatment)
                        f.write(f"{label}: {len(treatment_targets[treatment])} genes\n")
        
        # 4. Create consolidated summary
        summary_file = os.path.join(tf_analysis_dir, f"{'-'.join(tfs_of_interest).replace('(+)', '')}_analysis_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"Analysis Summary for {', '.join(tfs_of_interest)}\n")
            f.write("=" * 50 + "\n\n")
            
            # Summarize regulon presence across treatments
            f.write("1. Regulon Presence by Treatment\n")
            f.write("-" * 30 + "\n")
            for treatment in treatment_groups:
                if treatment in matched_regulons_by_treatment:
                    regulons = matched_regulons_by_treatment[treatment]
                    label = treatment_labels.get(treatment, treatment)
                    f.write(f"{label}: {len(regulons)} regulons - {', '.join([r.replace('(+)', '') for r in regulons])}\n")
                else:
                    label = treatment_labels.get(treatment, treatment)
                    f.write(f"{label}: No regulons found\n")
            
            f.write("\n")
            
            # Summarize activity patterns
            if len(all_matched_regulons) > 0 and len(activity_df.columns) > 0:
                f.write("2. Activity Patterns\n")
                f.write("-" * 30 + "\n")
                
                for regulon in sorted(all_matched_regulons):
                    clean_name = regulon.replace('(+)', '')
                    f.write(f"{clean_name}:\n")
                    
                    # Get values for this regulon across treatments
                    values = activity_df.loc[regulon].values
                    treatments = activity_df.columns.tolist()
                    
                    if np.sum(values) > 0:  # Only analyze if there's some activity
                        # Identify treatments with highest activity
                        max_idx = np.argmax(values)
                        max_treatment = treatments[max_idx]
                        max_value = values[max_idx]
                        max_label = treatment_labels.get(max_treatment, max_treatment)
                        
                        # Identify treatments with lowest activity
                        min_idx = np.argmin(values)
                        min_treatment = treatments[min_idx]
                        min_value = values[min_idx]
                        min_label = treatment_labels.get(min_treatment, min_treatment)
                        
                        # Calculate fold change
                        fold_change = max_value / max(min_value, 0.001)
                        
                        # Write summary
                        f.write(f"  Highest activity: {max_label} ({max_value:.3f})\n")
                        f.write(f"  Lowest activity: {min_label} ({min_value:.3f})\n")
                        f.write(f"  Max fold change: {fold_change:.2f}x\n")
                    else:
                        f.write("  No significant activity detected across treatments\n")
                    
                    f.write("\n")
        
        print("Enhanced TF-specific analysis complete!")
else:
    print("No TFs of interest specified. Skipping TF-specific analysis.")

# %% 11. Create a summary report
print("Creating summary report...")

with open(os.path.join(COMPARISON_DIR, "pyscenic_comparison_summary.txt"), 'w') as f:
    f.write("pySCENIC Cross-Treatment Comparison Summary\n")
    f.write("=========================================\n\n")
    
    # Dataset info
    f.write("Treatment Groups Analyzed:\n")
    for treatment in treatment_groups:
        label = treatment_labels.get(treatment, treatment)
        f.write(f"  - {treatment} ({label})\n")
    
    f.write("\n")
    
    # Regulon stats
    f.write("Regulon Statistics:\n")
    for treatment, regulons in regulons_dict.items():
        label = treatment_labels.get(treatment, treatment)
        f.write(f"  - {treatment} ({label}): {len(regulons)} regulons\n")
    
    if common_regulons:
        f.write(f"\nCommon Regulons (across all treatments): {len(common_regulons)}\n")
        f.write("Top 10 common regulons by average activity:\n")
        
        if avg_auc_df is not None:
            # Get average activity across all treatments for common regulons
            common_avg = avg_auc_df.loc[list(common_regulons)].mean(axis=1).sort_values(ascending=False)
            for i, (regulon, avg) in enumerate(common_avg.head(10).items()):
                f.write(f"  {i+1}. {regulon} (Avg AUC: {avg:.4f})\n")
    
    f.write("\n")
    
    # Treatment comparison summary
    f.write("Treatment Similarity Summary:\n")
    if 'similarity_matrix' in locals() and similarity_matrix is not None:
        # Get average similarity for each treatment
        avg_similarity = similarity_matrix.mean().sort_values(ascending=False)
        for treatment, avg_sim in avg_similarity.items():
            label = treatment_labels.get(treatment, treatment)
            f.write(f"  - {treatment} ({label}): Average similarity {avg_sim:.3f}\n")
        
        # Most similar pair
        most_similar = None
        highest_sim = 0
        
        for t1 in similarity_matrix.index:
            for t2 in similarity_matrix.columns:
                if t1 != t2 and similarity_matrix.loc[t1, t2] > highest_sim:
                    highest_sim = similarity_matrix.loc[t1, t2]
                    most_similar = (t1, t2)
        
        if most_similar:
            t1, t2 = most_similar
            label1 = treatment_labels.get(t1, t1)
            label2 = treatment_labels.get(t2, t2)
            f.write(f"\nMost similar treatments: {t1} ({label1}) and {t2} ({label2}) - Similarity: {highest_sim:.3f}\n")
    
    f.write("\n")
    
    # Highlight most distinctive treatments
    f.write("Most Distinctive Treatments (by number of enriched regulons):\n")
    for treatment, enriched_regs in sorted(treatment_enriched_regulons.items(), 
                                         key=lambda x: len(x[1]), reverse=True):
        label = treatment_labels.get(treatment, treatment)
        f.write(f"  - {treatment} ({label}): {len(enriched_regs)} enriched regulons\n")
    
    f.write("\n")
    
    # Top enriched regulons across treatments
    f.write("Top Enriched Regulons by Treatment:\n")
    for treatment, enriched_regs in treatment_enriched_regulons.items():
        if enriched_regs:
            label = treatment_labels.get(treatment, treatment)
            f.write(f"\n{treatment} ({label}):\n")
            for i, (regulon, fc) in enumerate(enriched_regs[:5]):  # Show top 5
                f.write(f"  {i+1}. {regulon} (Fold Change: {fc:.2f})\n")
    
    f.write("\n")
    
    # List of files generated
    f.write("Analysis Files Generated:\n")
    for root, dirs, files in os.walk(COMPARISON_DIR):
        rel_path = os.path.relpath(root, COMPARISON_DIR)
        if rel_path == '.':
            path_prefix = ''
        else:
            path_prefix = f"{rel_path}/"
            
        for file in files:
            if file != "pyscenic_comparison_summary.txt":  # Skip this summary file
                f.write(f"  - {path_prefix}{file}\n")

print("\nComparison analysis complete!")