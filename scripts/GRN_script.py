"""
Gene Regulatory Network Analysis and Interactive Visualization

This script analyzes gene regulatory networks from pySCENIC adjacency data
and creates comprehensive interactive visualizations with network statistics.

Original Author: Daniel Plaugher
Date: 6-7-25
NOTES: assistance cleaning and with HTML via Claude
Dependencies: pandas, networkx, pyvis, numpy
"""

#%% Libraries

import pandas as pd
import networkx as nx
from pyvis.network import Network
import os
import numpy as np

#%% Set up
# File path - update this to match your location
file_path = r"PATH-TO-adjacencies.csv"

# Read the adjacency file
df = pd.read_csv(file_path)

# Sort by importance (descending) and get top 500
top_connections = df.sort_values('importance', ascending=False).head(500)

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges with weights
for _, row in top_connections.iterrows():
    G.add_edge(row['TF'], row['target'], weight=row['importance'])

#%%  Network Analysis
# Basic Network Composition Analysis
print("=" * 60)
print("BASIC NETWORK COMPOSITION")
print("=" * 60)

# Identify TF and target nodes, recognizing dual-role genes
tf_nodes = set(top_connections['TF'])
target_nodes = set(top_connections['target'])
dual_role_nodes = tf_nodes.intersection(target_nodes)

print(f"Total unique genes: {len(tf_nodes.union(target_nodes))}")
print(f"Genes acting as TF only: {len(tf_nodes - dual_role_nodes)}")
print(f"Genes acting as target only: {len(target_nodes - dual_role_nodes)}")
print(f"Genes with dual role (both TF and target): {len(dual_role_nodes)}")
print(f"Total edges: {G.number_of_edges()}")

if len(dual_role_nodes) > 0:
    print(f"Examples of dual-role genes: {list(dual_role_nodes)[:5]}")
    
#=============================================================================
# Connectivity Analysis
print("\n" + "=" * 60)
print("NODE CONNECTIVITY ANALYSIS")
print("=" * 60)

# Calculate degree statistics
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())
total_degrees = dict(G.degree())

# Get connectivity statistics
print(f"Average in-degree: {np.mean(list(in_degrees.values())):.2f}")
print(f"Average out-degree: {np.mean(list(out_degrees.values())):.2f}")
print(f"Average total degree: {np.mean(list(total_degrees.values())):.2f}")

# Find highly connected nodes (hubs)
print(f"\nMax in-degree: {max(in_degrees.values())}")
print(f"Max out-degree: {max(out_degrees.values())}")

# Top 5 nodes by in-degree (most regulated genes)
top_targets = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\nTop 5 most regulated genes (highest in-degree):")
for gene, degree in top_targets:
    print(f"  {gene}: {degree} incoming connections")

# Top 5 nodes by out-degree (master regulators)
top_regulators = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\nTop 5 master regulators (highest out-degree):")
for gene, degree in top_regulators:
    print(f"  {gene}: {degree} outgoing connections")

#=============================================================================
# Subnetwork Analysis
print("\n" + "=" * 60)
print("SUBNETWORK ANALYSIS")
print("=" * 60)

# For directed graphs, we can look at:
# 1. Weakly connected components (ignoring edge direction)
# 2. Strongly connected components (following edge direction)

# Weakly connected components
weak_components = list(nx.weakly_connected_components(G))
print(f"Number of weakly connected components: {len(weak_components)}")

if len(weak_components) > 1:
    component_sizes = [len(comp) for comp in weak_components]
    print(f"Component sizes: {sorted(component_sizes, reverse=True)}")
    
    # Show details of smaller components
    for i, comp in enumerate(weak_components):
        if len(comp) < 10:  # Only show small components
            print(f"  Component {i+1} ({len(comp)} nodes): {list(comp)}")
else:
    print("Network is fully connected (weakly)")

# Strongly connected components
strong_components = list(nx.strongly_connected_components(G))
print(f"\nNumber of strongly connected components: {len(strong_components)}")

# Filter out single-node components for meaningful analysis
multi_node_strong = [comp for comp in strong_components if len(comp) > 1]
print(f"Strongly connected components with >1 node: {len(multi_node_strong)}")

if multi_node_strong:
    for i, comp in enumerate(multi_node_strong):
        print(f"  Strong component {i+1} ({len(comp)} nodes): {list(comp)}")

#=============================================================================
# Network Structure Analysis
print("\n" + "=" * 60)
print("NETWORK STRUCTURE ANALYSIS")
print("=" * 60)

# Network density
density = nx.density(G)
print(f"Network density: {density:.4f}")
print(f"  (Out of {G.number_of_nodes()} * {G.number_of_nodes()-1} = {G.number_of_nodes()*(G.number_of_nodes()-1)} possible edges)")

# Reciprocity - how many edges have reciprocal connections
reciprocity = nx.reciprocity(G)
print(f"Reciprocity: {reciprocity:.4f}")
print(f"  (Fraction of edges that are bidirectional)")

# Find reciprocal edges
reciprocal_edges = []
for edge in G.edges():
    if G.has_edge(edge[1], edge[0]):  # Check if reverse edge exists
        reciprocal_edges.append(edge)

print(f"Number of reciprocal edge pairs: {len(reciprocal_edges)//2}")

#=============================================================================
# Centrality Analysis
print("\n" + "=" * 60)
print("CENTRALITY ANALYSIS")
print("=" * 60)

# Calculate different centrality measures
betweenness = nx.betweenness_centrality(G)
closeness_in = nx.closeness_centrality(G.reverse())  # Incoming paths
closeness_out = nx.closeness_centrality(G)  # Outgoing paths

# Top nodes by betweenness centrality (bridge nodes)
top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 bridge nodes (highest betweenness centrality):")
for gene, centrality in top_betweenness:
    print(f"  {gene}: {centrality:.4f}")

# Top nodes by closeness centrality (incoming)
top_closeness_in = sorted(closeness_in.items(), key=lambda x: x[1], reverse=True)[:5]
print(f"\nTop 5 nodes by incoming closeness centrality:")
for gene, centrality in top_closeness_in:
    print(f"  {gene}: {centrality:.4f}")

#=============================================================================
# Edge Weight Analysis
print("\n" + "=" * 60)
print("EDGE WEIGHT ANALYSIS")
print("=" * 60)

weights = [G[u][v]['weight'] for u, v in G.edges()]
print(f"Edge weight statistics:")
print(f"  Min weight: {min(weights):.4f}")
print(f"  Max weight: {max(weights):.4f}")
print(f"  Mean weight: {np.mean(weights):.4f}")
print(f"  Median weight: {np.median(weights):.4f}")
print(f"  Std deviation: {np.std(weights):.4f}")

# Find strongest connections
edge_weights = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
strongest_edges = sorted(edge_weights, key=lambda x: x[2], reverse=True)[:5]
print(f"\nTop 5 strongest connections:")
for tf, target, weight in strongest_edges:
    print(f"  {tf} -> {target}: {weight:.4f}")

#=============================================================================
# Network Motifs (Simple patterns)
print("\n" + "=" * 60)
print("SIMPLE NETWORK PATTERNS")
print("=" * 60)

# Count simple patterns
self_loops = list(nx.nodes_with_selfloops(G))
print(f"Self-loops (auto-regulation): {len(self_loops)}")
if self_loops:
    print(f"  Self-regulating genes: {self_loops[:5]}")

# Find nodes that regulate each other (mutual regulation)
mutual_pairs = []
for node1 in G.nodes():
    for node2 in G.successors(node1):
        if G.has_edge(node2, node1) and node1 < node2:  # Avoid duplicates
            mutual_pairs.append((node1, node2))

print(f"\nMutual regulation pairs: {len(mutual_pairs)}")
if mutual_pairs:
    print(f"  Examples: {mutual_pairs[:5]}")

#%% Pyvis Visualization

print("\n" + "=" * 60)
print("CREATING INTERACTIVE VISUALIZATION")
print("=" * 60)

# Create pyvis network with configuration enabled for interactive changes
net = Network(height="900px", width="100%", notebook=False, directed=True)
net.toggle_physics(True)
net.show_buttons(filter_=['physics'])

# Prepare edge data
edge_data = {}
for _, row in top_connections.iterrows():
    edge_data[(row['TF'], row['target'])] = row['importance']

# Add nodes with enhanced tooltips showing connectivity info
for node_id in G.nodes():
    in_deg = in_degrees[node_id]
    out_deg = out_degrees[node_id]
    
    if node_id in dual_role_nodes:
        tooltip = f"Dual Role Gene: {node_id}<br/>In-degree: {in_deg}<br/>Out-degree: {out_deg}"
        net.add_node(node_id, color='#9C27B0', size=35, title=tooltip,
                    font={'size': 24, 'face': 'arial', 'color': 'black'})
    elif node_id in tf_nodes:
        tooltip = f"Transcription Factor: {node_id}<br/>Out-degree: {out_deg}"
        net.add_node(node_id, color='#4285F4', size=30, title=tooltip,
                    font={'size': 22, 'face': 'arial', 'color': 'black'})
    else:
        tooltip = f"Target Gene: {node_id}<br/>In-degree: {in_deg}"
        net.add_node(node_id, color='#34A853', size=25, title=tooltip,
                    font={'size': 20, 'face': 'arial', 'color': 'black'})

# Calculate min and max weights for normalization
min_weight = min(edge_data.values())
max_weight = max(edge_data.values())

# Add edges with proper weights
for (source, target), weight in edge_data.items():
    normalized_width = ((weight - min_weight) / (max_weight - min_weight)) * 10 + 2
    net.add_edge(source=source, to=target, width=normalized_width, 
                title=f"Importance: {weight:.4f}",
                arrows={'to': {'enabled': True, 'scaleFactor': 1.0}})
#%% HTML options
# Set network options 
net.set_options("""
{
  "physics": {
    "enabled": true,
    "stabilization": {
      "iterations": 200,
      "updateInterval": 25
    },
    "barnesHut": {
      "gravitationalConstant": -2000,
      "centralGravity": 0.1,
      "springLength": 200,
      "springConstant": 0.04,
      "damping": 0.09,
      "avoidOverlap": 0.2
    }
  },
  "edges": {
    "smooth": {
      "type": "dynamic",
      "forceDirection": "none"
    },
    "color": {
      "inherit": false
    },
    "font": {
      "size": 14
    }
  },
  "nodes": {
    "shape": "dot",
    "scaling": {
      "label": {
        "enabled": true,
        "min": 18,
        "max": 26
      }
    },
    "shadow": true
  },
  "interaction": {
    "navigationButtons": true,
    "keyboard": true,
    "hover": true,
    "multiselect": true,
    "hideEdgesOnDrag": false,
    "tooltipDelay": 100
  },
  "configure": {
    "enabled": true,
    "filter": "physics,layout,interaction,manipulation,renderer",
    "showButton": true
  },
  "layout": {
    "improvedLayout": true,
    "hierarchical": {
      "enabled": false
    }
  }
}
""")

# Enhanced HTML with additional statistics
html_head = """
<style>
  body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 20px;
    background-color: #f5f5f5;
  }
  .header {
    text-align: center;
    margin-bottom: 20px;
  }
  .container {
    display: flex;
    flex-direction: column;
    max-width: 1200px;
    margin: 0 auto;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    padding: 20px;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-bottom: 20px;
    padding: 15px;
    background-color: #f9f9f9;
    border-radius: 8px;
  }
  .control-group {
    margin-bottom: 10px;
  }
  label {
    font-weight: bold;
    margin-right: 10px;
  }
  button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 8px 16px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 14px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 4px;
  }
  button:hover {
    background-color: #45a049;
  }
  select, input {
    padding: 6px;
    border-radius: 4px;
    border: 1px solid #ddd;
  }
  .legend {
    display: flex;
    justify-content: center;
    margin-top: 15px;
    flex-wrap: wrap;
    gap: 20px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    margin-right: 15px;
  }
  .legend-color {
    width: 20px;
    height: 20px;
    border-radius: 50%;
    margin-right: 8px;
  }
  .stats {
    margin-top: 15px;
    font-size: 14px;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 4px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 10px;
  }
  .stat-section {
    background-color: white;
    padding: 10px;
    border-radius: 4px;
    border-left: 4px solid #4CAF50;
  }
  .stat-section h4 {
    margin-top: 0;
    color: #333;
  }
</style>

<div class="container">
  <div class="header">
    <h1>Gene Regulatory Network Analysis</h1>
    <p>Interactive visualization with comprehensive network statistics</p>
  </div>
  
  <div class="controls">
    <div class="control-group">
      <label for="layout-select">Layout:</label>
      <select id="layout-select" onchange="changeLayout()">
        <option value="standard">Force-Directed</option>
        <option value="hierarchical">Hierarchical</option>
      </select>
    </div>
    
    <div class="control-group">
      <label for="node-size">Node Size:</label>
      <input type="range" id="node-size" min="10" max="50" value="30" onchange="changeNodeSize()">
    </div>
    
    <div class="control-group">
      <label for="font-size">Font Size:</label>
      <input type="range" id="font-size" min="10" max="30" value="22" onchange="changeFontSize()">
    </div>
    
    <div class="control-group">
      <button onclick="stabilizeNetwork()">Stabilize Network</button>
      <button onclick="resetNetwork()">Reset View</button>
      <button onclick="exportNetwork()">Export as PNG</button>
    </div>
  </div>
  
  <div class="legend">
    <div class="legend-item">
      <div class="legend-color" style="background-color: #4285F4;"></div>
      <span>Transcription Factor</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background-color: #34A853;"></div>
      <span>Target Gene</span>
    </div>
    <div class="legend-item">
      <div class="legend-color" style="background-color: #9C27B0;"></div>
      <span>Dual Role (TF & Target)</span>
    </div>
  </div>
  
  <div id="stats" class="stats">
    <div class="stat-section">
      <h4>Basic Statistics</h4>
      <p>Nodes: <span id="node-count"></span></p>
      <p>Edges: <span id="edge-count"></span></p>
      <p>Density: <span id="density"></span></p>
    </div>
    
    <div class="stat-section">
      <h4>Node Types</h4>
      <p>TFs only: <span id="tf-count"></span></p>
      <p>Targets only: <span id="target-count"></span></p>
      <p>Dual role: <span id="dual-count"></span></p>
    </div>
    
    <div class="stat-section">
      <h4>Connectivity</h4>
      <p>Avg in-degree: <span id="avg-in-degree"></span></p>
      <p>Avg out-degree: <span id="avg-out-degree"></span></p>
      <p>Max in-degree: <span id="max-in-degree"></span></p>
    </div>
  </div>
</div>

<script>
  function changeLayout() {
    const layout = document.getElementById('layout-select').value;
    if (layout === 'hierarchical') {
      network.setOptions({
        layout: {
          hierarchical: {
            enabled: true,
            direction: 'UD',
            sortMethod: 'directed',
            nodeSpacing: 150,
            levelSeparation: 200
          }
        }
      });
    } else {
      network.setOptions({
        layout: {
          hierarchical: {
            enabled: false
          }
        }
      });
      network.stabilize(100);
    }
  }
  
  function changeNodeSize() {
    const size = parseInt(document.getElementById('node-size').value);
    const nodes = network.body.data.nodes.get();
    nodes.forEach(node => {
      if (node.color === '#9C27B0') {
        node.size = size + 5;
      } else if (node.color === '#4285F4') {
        node.size = size;
      } else {
        node.size = size - 5;
      }
    });
    network.body.data.nodes.update(nodes);
  }
  
  function changeFontSize() {
    const size = parseInt(document.getElementById('font-size').value);
    const nodes = network.body.data.nodes.get();
    nodes.forEach(node => {
      node.font.size = size;
    });
    network.body.data.nodes.update(nodes);
  }
  
  function stabilizeNetwork() {
    network.stabilize(500);
  }
  
  function resetNetwork() {
    network.fit({
      animation: {
        duration: 1000,
        easingFunction: 'easeInOutQuad'
      }
    });
  }
  
  function exportNetwork() {
    const canvas = document.querySelector('canvas');
    const link = document.createElement('a');
    link.href = canvas.toDataURL('image/png');
    link.download = 'gene_regulatory_network.png';
    link.click();
  }
  
  // Update statistics when the page loads
  window.addEventListener('load', () => {
    setTimeout(() => {
      if (typeof network !== 'undefined') {
        const nodes = network.body.data.nodes.get();
        const edges = network.body.data.edges.get();
        
        document.getElementById('node-count').textContent = nodes.length;
        document.getElementById('edge-count').textContent = edges.length;
        
        // Calculate density
        const maxEdges = nodes.length * (nodes.length - 1);
        const density = (edges.length / maxEdges * 100).toFixed(2);
        document.getElementById('density').textContent = density + '%';
        
        // Count node types and calculate connectivity
        let tfCount = 0, targetCount = 0, dualCount = 0;
        let totalInDegree = 0, totalOutDegree = 0, maxInDegree = 0;
        
        nodes.forEach(node => {
          const inDegree = edges.filter(e => e.to === node.id).length;
          const outDegree = edges.filter(e => e.from === node.id).length;
          
          totalInDegree += inDegree;
          totalOutDegree += outDegree;
          maxInDegree = Math.max(maxInDegree, inDegree);
          
          if (node.color === '#9C27B0') dualCount++;
          else if (node.color === '#4285F4') tfCount++;
          else targetCount++;
        });
        
        document.getElementById('tf-count').textContent = tfCount;
        document.getElementById('target-count').textContent = targetCount;
        document.getElementById('dual-count').textContent = dualCount;
        document.getElementById('avg-in-degree').textContent = (totalInDegree / nodes.length).toFixed(2);
        document.getElementById('avg-out-degree').textContent = (totalOutDegree / nodes.length).toFixed(2);
        document.getElementById('max-in-degree').textContent = maxInDegree;
      }
    }, 1000);
  });
</script>
"""

#%% Save the visualization
output_dir = os.path.dirname(file_path)
output_path = os.path.join(output_dir, "enhanced_gene_regulatory_network.html")
net.save_graph(output_path)

# Insert the custom HTML
with open(output_path, 'r', encoding='utf-8') as file:
    html_content = file.read()

with open(output_path, 'w', encoding='utf-8') as file:
    html_content = html_content.replace('<body>', f'<body>\n{html_head}')
    file.write(html_content)

print(f"\nInteractive network visualization saved to: {output_path}")

#%%
# ============================================================================
# SAVE ANALYSIS RESULTS TO FILE
# ============================================================================

# Create output file path
results_file = os.path.join(output_dir, f"network_analysis_results_{os.path.splitext(os.path.basename(file_path))[0]}.txt")

with open(results_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("GENE REGULATORY NETWORK ANALYSIS RESULTS\n")
    f.write("=" * 60 + "\n\n")
    
    # Basic composition
    f.write("BASIC NETWORK COMPOSITION\n")
    f.write("-" * 30 + "\n")
    f.write(f"Total unique genes: {len(tf_nodes.union(target_nodes))}\n")
    f.write(f"Genes acting as TF only: {len(tf_nodes - dual_role_nodes)}\n")
    f.write(f"Genes acting as target only: {len(target_nodes - dual_role_nodes)}\n")
    f.write(f"Genes with dual role: {len(dual_role_nodes)}\n")
    f.write(f"Total edges: {G.number_of_edges()}\n")
    if dual_role_nodes:
        f.write(f"All dual-role genes: {', '.join(list(dual_role_nodes))}\n")
    
    # Connectivity
    f.write(f"\nCONNECTIVITY ANALYSIS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Average in-degree: {np.mean(list(in_degrees.values())):.2f}\n")
    f.write(f"Average out-degree: {np.mean(list(out_degrees.values())):.2f}\n")
    f.write(f"Max in-degree: {max(in_degrees.values())}\n")
    f.write(f"Max out-degree: {max(out_degrees.values())}\n")
    
    f.write(f"\nTop 5 most regulated genes:\n")
    for gene, degree in top_targets:
        f.write(f"  {gene}: {degree} connections\n")
    
    f.write(f"\nTop 5 master regulators:\n")
    for gene, degree in top_regulators:
        f.write(f"  {gene}: {degree} connections\n")
    
    # Subnetworks
    f.write(f"\nSUBNETWORK ANALYSIS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Weakly connected components: {len(weak_components)}\n")
    if len(weak_components) > 1:
        component_sizes = [len(comp) for comp in weak_components]
        f.write(f"Component sizes: {sorted(component_sizes, reverse=True)}\n")
    
    f.write(f"Strongly connected components: {len(strong_components)}\n")
    f.write(f"Multi-node strong components: {len(multi_node_strong)}\n")
    
    # Write all multi-node strong components
    if multi_node_strong:
        f.write(f"All multi-node strong components:\n")
        for i, comp in enumerate(multi_node_strong):
            f.write(f"  Component {i+1} ({len(comp)} nodes): {list(comp)}\n")
    
    # Structure
    f.write(f"\nNETWORK STRUCTURE\n")
    f.write("-" * 30 + "\n")
    f.write(f"Network density: {density:.4f}\n")
    f.write(f"Reciprocity: {reciprocity:.4f}\n")
    f.write(f"Reciprocal edge pairs: {len(reciprocal_edges)//2}\n")
    
    # Centrality
    f.write(f"\nCENTRALITY ANALYSIS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Top 5 bridge nodes (betweenness centrality):\n")
    for gene, centrality in top_betweenness:
        f.write(f"  {gene}: {centrality:.4f}\n")
    
    # Weights
    f.write(f"\nEDGE WEIGHT STATISTICS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Min weight: {min(weights):.4f}\n")
    f.write(f"Max weight: {max(weights):.4f}\n")
    f.write(f"Mean weight: {np.mean(weights):.4f}\n")
    f.write(f"Median weight: {np.median(weights):.4f}\n")
    f.write(f"Std deviation: {np.std(weights):.4f}\n")
    
    f.write(f"\nTop 5 strongest connections:\n")
    for tf, target, weight in strongest_edges:
        f.write(f"  {tf} -> {target}: {weight:.4f}\n")
    
    # Patterns
    f.write(f"\nNETWORK PATTERNS\n")
    f.write("-" * 30 + "\n")
    f.write(f"Self-loops (auto-regulation): {len(self_loops)}\n")
    if self_loops:
        f.write(f"Self-regulating genes: {', '.join(self_loops)}\n")
    f.write(f"Mutual regulation pairs: {len(mutual_pairs)}\n")
    if mutual_pairs:
        f.write(f"All mutual regulation pairs: {str(mutual_pairs)}\n")

print(f"Analysis results saved to: {results_file}")
print("\nAnalysis complete!")