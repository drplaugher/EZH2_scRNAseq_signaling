#' Module Score Analysis for Single-Cell RNA-seq Data
#' Original code by Daniel Plaugher, cleaned and documented 6-7-25
#'
#' Generates module scores for gene sets and creates visualizations with 
#' statistical comparisons between groups.
#'
#' @param seurat_obj Seurat object containing single-cell RNA-seq data
#' @param marker_genes Character vector of positive marker genes
#' @param require_all Logical, whether all markers must be expressed (currently unused)
#' @param group_var Character, column name in metadata for grouping cells
#' @param desired_order Character vector specifying group order for plots
#' @param output_prefix Character, prefix for output files and plot titles
#' @param save_results Logical, whether to save plots and statistics to disk
#' @param output_dir Character, directory path for saving results
#' @param perform_neg_markers Logical, whether to include negative markers
#' @param neg_markers Character vector of negative marker genes
#' @param plot_type Character vector, plot types: "violin", "boxplot", "dotplot"
#' @param compute_pairwise Logical, whether to compute pairwise comparisons
#' @param hide_ns Logical, whether to hide non-significant comparisons
#' @param pt_size Numeric, point size for violin plots
#' @param plot_width Numeric, width of saved plots in inches
#' @param plot_height Numeric, height of saved plots in inches
#' @param return_seurat Logical, whether to return the modified Seurat object
#'
#' @return List containing plots, summary statistics, and optionally the Seurat object
#' @export

module_builder <- function(
    seurat_obj,
    marker_genes,
    require_all = FALSE,
    group_var = NULL,
    desired_order = NULL,
    output_prefix = "Population",
    save_results = FALSE,
    output_dir = "Module_Analysis",
    perform_neg_markers = FALSE,
    neg_markers = NULL,
    plot_type = c("violin", "boxplot", "dotplot"),
    compute_pairwise = TRUE,
    hide_ns = TRUE,
    pt_size = 0.1,
    plot_width = 12,
    plot_height = 8,
    return_seurat = TRUE
) {
  
  # Load required libraries
  suppressPackageStartupMessages({
    require(Seurat)
    require(dplyr)
    require(ggplot2)
    require(patchwork)
    require(ggpubr)
  })
  
  # Initialize results list
  results <- list()
  
  # Store parameters
  results$parameters <- list(
    marker_genes = marker_genes,
    require_all = require_all,
    group_var = group_var,
    desired_order = desired_order,
    output_prefix = output_prefix
  )
  
  # Create output directory if saving results
  if (save_results) {
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
      message("Created output directory: ", output_dir)
    }
  }
  
  # Set RNA as default assay
  DefaultAssay(seurat_obj) <- "RNA"
  
  # Apply custom ordering to groups if provided
  if (!is.null(group_var) && !is.null(desired_order)) {
    if (all(desired_order %in% unique(seurat_obj@meta.data[[group_var]]))) {
      seurat_obj@meta.data[[group_var]] <- factor(
        seurat_obj@meta.data[[group_var]], 
        levels = desired_order
      )
      message("Applied custom ordering to group variable")
    } else {
      warning("Some values in desired_order don't match the data in group_var")
      results$warnings <- c(results$warnings, "Order mismatch")
    }
  }
  
  # Validate marker genes exist in dataset
  seurat_genes <- rownames(seurat_obj)
  missing_genes <- setdiff(marker_genes, seurat_genes)
  
  if (length(missing_genes) > 0) {
    warning("The following genes were not found in the Seurat object: ",
            paste(missing_genes, collapse = ", "))
    results$missing_genes <- missing_genes
    
    # Remove missing genes from marker list
    marker_genes <- intersect(marker_genes, seurat_genes)
    if (length(marker_genes) == 0) {
      stop("No marker genes found in the Seurat object")
    }
  }
  
  # Handle negative markers
  if (perform_neg_markers && is.null(neg_markers)) {
    warning("perform_neg_markers is TRUE but no neg_markers provided")
    perform_neg_markers <- FALSE
  }
  
  if (perform_neg_markers) {
    missing_neg_genes <- setdiff(neg_markers, seurat_genes)
    if (length(missing_neg_genes) > 0) {
      warning("The following negative marker genes were not found: ",
              paste(missing_neg_genes, collapse = ", "))
      results$missing_neg_genes <- missing_neg_genes
      
      # Remove missing genes from negative marker list
      neg_markers <- intersect(neg_markers, seurat_genes)
    }
  }
  
  # Create marker list for AddModuleScore
  marker_list <- list(marker_genes)
  names(marker_list) <- output_prefix
  
  if (perform_neg_markers && length(neg_markers) > 0) {
    marker_list[[paste0(output_prefix, "-Neg")]] <- neg_markers
  }
  
  # Compute module scores
  message("Computing module scores for ", output_prefix)
  seurat_obj <- AddModuleScore(
    seurat_obj, 
    features = marker_list, 
    name = paste0(output_prefix, "_Score")
  )
  
  # Store column names for easy reference
  module_score_column <- paste0(output_prefix, "_Score1")
  results$module_score_column <- module_score_column
  
  # Compute combined score if negative markers are used
  if (perform_neg_markers && length(neg_markers) > 0) {
    neg_score_column <- paste0(output_prefix, "_Score2")
    combined_score_column <- paste0(output_prefix, "_Combined")
    
    seurat_obj@meta.data[[combined_score_column]] <- 
      seurat_obj@meta.data[[module_score_column]] - seurat_obj@meta.data[[neg_score_column]]
    
    results$neg_score_column <- neg_score_column
    results$combined_score_column <- combined_score_column
  }
  
  # Validate plot types
  plot_types <- match.arg(plot_type, c("violin", "boxplot", "dotplot"), several.ok = TRUE)
  results$plots <- list()
  
  # Generate plots if group variable is provided
  if (!is.null(group_var)) {
    
    # Create dotplot
    if ("dotplot" %in% plot_types) {
      message("Generating dot plot")
      
      dot_plot <- DotPlot(seurat_obj, features = marker_genes, group.by = group_var) + 
        RotatedAxis() +
        ggtitle(paste(output_prefix, "Marker Expression")) +
        theme(plot.title = element_text(hjust = 0.5))
      
      results$plots$dotplot <- dot_plot
      
      if (save_results) {
        file_name <- file.path(output_dir, paste0(output_prefix, "_markers_dotplot.png"))
        ggsave(file_name, plot = dot_plot, width = plot_width, height = plot_height, dpi = 300)
        message("Saved dotplot to ", file_name)
      }
    }
    
    # Create violin plots
    if ("violin" %in% plot_types) {
      message("Generating violin plot")
      
      # Main violin plot
      vln_plot <- VlnPlot(seurat_obj, features = module_score_column, 
                          group.by = group_var, pt.size = pt_size) +
        ggtitle(paste(output_prefix, "Module Score")) +
        theme(
          plot.title = element_text(hjust = 0.5),
          axis.text.x = element_text(angle = 45, hjust = 1)
        )
      
      results$plots$violin <- vln_plot
      
      if (save_results) {
        file_name <- file.path(output_dir, paste0(output_prefix, "_score_violin.png"))
        ggsave(file_name, plot = vln_plot, width = plot_width, height = plot_height, dpi = 300)
        message("Saved violin plot to ", file_name)
      }
      
      # Combined score violin plot if negative markers were used
      if (perform_neg_markers && length(neg_markers) > 0) {
        comb_vln_plot <- VlnPlot(seurat_obj, features = combined_score_column, 
                                 group.by = group_var, pt.size = pt_size) +
          ggtitle(paste(output_prefix, "Combined Score (Pos-Neg)")) +
          theme(
            plot.title = element_text(hjust = 0.5),
            axis.text.x = element_text(angle = 45, hjust = 1)
          )
        
        results$plots$violin_combined <- comb_vln_plot
        
        if (save_results) {
          file_name <- file.path(output_dir, paste0(output_prefix, "_combined_score_violin.png"))
          ggsave(file_name, plot = comb_vln_plot, width = plot_width, height = plot_height, dpi = 300)
          message("Saved combined violin plot to ", file_name)
        }
      }
    }
    
    # Create boxplots
    if ("boxplot" %in% plot_types) {
      message("Generating boxplot")
      
      # Create pairwise comparisons if requested
      if (compute_pairwise) {
        group_levels <- levels(factor(seurat_obj@meta.data[[group_var]]))
        comparisons <- combn(group_levels, 2, simplify = FALSE)
      } else {
        comparisons <- NULL
      }
      
      # Function to create boxplot with statistical comparisons
      create_boxplot <- function(score_column, title) {
        meta_data <- seurat_obj@meta.data
        
        p <- ggplot(meta_data, aes(x = .data[[group_var]], y = .data[[score_column]], 
                                   fill = .data[[group_var]])) +
          geom_boxplot() +
          stat_summary(fun = mean, geom = "text", 
                       aes(label = round(after_stat(y), 2)),
                       vjust = -0.5, size = 4, fontface = "bold", color = "black") +
          theme_minimal() +
          labs(title = title, y = "Module Score", x = group_var) +
          theme(
            text = element_text(size = 12), 
            axis.text.x = element_text(angle = 45, hjust = 1),
            plot.title = element_text(hjust = 0.5)
          )
        
        # Add statistical comparisons if requested
        if (compute_pairwise && !is.null(comparisons)) {
          p <- p + stat_compare_means(
            comparisons = comparisons, 
            method = "wilcox.test",
            label = "p.signif", 
            size = 4, 
            hide.ns = hide_ns, 
            vjust = 1
          )
        }
        
        return(p)
      }
      
      # Main boxplot
      box_plot <- create_boxplot(module_score_column, 
                                 paste(output_prefix, "Score by", group_var))
      results$plots$boxplot <- box_plot
      
      if (save_results) {
        file_name <- file.path(output_dir, paste0(output_prefix, "_score_boxplot.png"))
        ggsave(file_name, plot = box_plot, width = plot_width, height = plot_height, dpi = 300)
        message("Saved boxplot to ", file_name)
      }
      
      # Combined score boxplot if negative markers were used
      if (perform_neg_markers && length(neg_markers) > 0) {
        comb_box_plot <- create_boxplot(combined_score_column, 
                                        paste(output_prefix, "Combined Score (Pos-Neg) by", group_var))
        results$plots$boxplot_combined <- comb_box_plot
        
        if (save_results) {
          file_name <- file.path(output_dir, paste0(output_prefix, "_combined_score_boxplot.png"))
          ggsave(file_name, plot = comb_box_plot, width = plot_width, height = plot_height, dpi = 300)
          message("Saved combined boxplot to ", file_name)
        }
      }
    }
  }
  
  # Calculate summary statistics
  if (!is.null(group_var)) {
    message("Calculating summary statistics")
    
    if (perform_neg_markers && length(neg_markers) > 0) {
      summary_stats <- seurat_obj@meta.data %>%
        group_by(.data[[group_var]]) %>%
        summarise(
          Mean_Score = mean(.data[[module_score_column]], na.rm = TRUE),
          SD_Score = sd(.data[[module_score_column]], na.rm = TRUE),
          Mean_Neg_Score = mean(.data[[neg_score_column]], na.rm = TRUE),
          Mean_Combined_Score = mean(.data[[combined_score_column]], na.rm = TRUE),
          Cell_Count = n()
        )
    } else {
      summary_stats <- seurat_obj@meta.data %>%
        group_by(.data[[group_var]]) %>%
        summarise(
          Mean_Score = mean(.data[[module_score_column]], na.rm = TRUE),
          SD_Score = sd(.data[[module_score_column]], na.rm = TRUE),
          Cell_Count = n()
        )
    }
    
    results$summary_stats <- summary_stats
    
    # Save summary statistics
    if (save_results) {
      file_name <- file.path(output_dir, paste0(output_prefix, "_summary_stats.csv"))
      write.csv(summary_stats, file = file_name, row.names = FALSE)
      message("Saved summary statistics to ", file_name)
    }
  }
  
  # Save modified Seurat object if requested
  if (save_results) {
    file_name <- file.path(output_dir, paste0(output_prefix, "_seurat_object.rds"))
    saveRDS(seurat_obj, file = file_name)
    message("Saved modified Seurat object to ", file_name)
  }
  
  # Add Seurat object to results if requested
  if (return_seurat) {
    results$seurat_obj <- seurat_obj
  }
  
  return(results)
}