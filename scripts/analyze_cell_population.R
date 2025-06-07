#' Analyze Cell Population Proportions in Single-Cell RNA-seq Data
#' Original code by Daniel Plaugher, cleaned and documented 6-7-25
#'
#' Identifies cells expressing specific marker genes and compares proportions
#' between treatment groups using Fisher's exact test.
#'
#' @param seurat_obj Seurat object containing single-cell RNA-seq data
#' @param marker_genes Character vector of marker genes to identify cell population
#' @param require_all Logical, whether cells must express ALL markers (TRUE) or ANY marker (FALSE)
#' @param group_var Character, column name in metadata for grouping (default: "resistance")
#' @param desired_order Character vector specifying group order for plots and tables
#' @param output_prefix Character, prefix for output files and plot titles
#' @param save_results Logical, whether to save results to files
#'
#' @return List containing:
#'   \item{seurat_object}{Modified Seurat object with population identification}
#'   \item{summary}{Basic summary dataframe}
#'   \item{detailed_summary}{Detailed summary with counts and percentages}
#'   \item{statistical_tests}{Pairwise Fisher's exact test results}
#'   \item{plot}{ggplot object showing proportions}
#'
#' @export
analyze_cell_population <- function(seurat_obj, 
                                    marker_genes, 
                                    require_all = TRUE,
                                    group_var = "resistance",
                                    desired_order = NULL,
                                    output_prefix = "cell_population",
                                    save_results = TRUE) {
  
  # Load required libraries
  suppressPackageStartupMessages({
    require(Seurat)
    require(ggplot2)
    require(dplyr)
  })
  
  # Extract expression data for markers
  expression_data <- FetchData(seurat_obj, vars = marker_genes)
  
  # Debug information
  cat("Expression data dimensions:", dim(expression_data), "\n")
  
  # Identify positive cells based on criteria
  if (require_all) {
    # Cells must express ALL markers
    positive_cells <- rowSums(expression_data > 0) == length(marker_genes)
  } else {
    # Cells must express ANY marker
    positive_cells <- rowSums(expression_data > 0) >= 1
  }
  
  cat("Number of positive cells:", sum(positive_cells), "\n")
  
  # Add population identification to metadata
  metadata_name <- paste0(output_prefix, "_Positive")
  seurat_obj[[metadata_name]] <- positive_cells
  
  # Get group variable - ensure proper ordering
  groups <- FetchData(seurat_obj, vars = group_var)[, 1]
  
  # Validate data consistency
  if (length(positive_cells) != length(groups)) {
    stop("Length mismatch: positive_cells (", length(positive_cells), 
         ") and groups (", length(groups), ") have different lengths")
  }
  
  # Create cross-tabulation
  cross_tab <- table(positive_cells, groups)
  
  # Extract positive cell counts by group
  if ("TRUE" %in% rownames(cross_tab)) {
    positive_counts <- cross_tab["TRUE", ]
  } else {
    positive_counts <- rep(0, ncol(cross_tab))
    names(positive_counts) <- colnames(cross_tab)
  }
  
  # Calculate total cells and proportions
  total_counts <- colSums(cross_tab)
  proportions <- positive_counts / total_counts
  
  # Create summary dataframe
  summary_df <- data.frame(
    Group = names(proportions), 
    Proportion = as.vector(proportions)
  )
  
  # Apply custom ordering if specified
  if (!is.null(desired_order)) {
    valid_order <- desired_order[desired_order %in% summary_df$Group]
    if (length(valid_order) > 0) {
      summary_df <- summary_df[match(valid_order, summary_df$Group), ]
      summary_df <- summary_df[!is.na(summary_df$Group), ]
    } else {
      warning("None of the values in desired_order match group names")
    }
  }
  
  # Prepare groups for statistical comparisons
  groups_for_comparison <- unique(groups)
  if (!is.null(desired_order)) {
    valid_order <- desired_order[desired_order %in% groups_for_comparison]
    if (length(valid_order) > 0) {
      groups_for_comparison <- valid_order
    }
  }
  
  # Generate pairwise comparisons
  comparisons <- combn(groups_for_comparison, 2, simplify = FALSE)
  
  # Create data for statistical testing
  stat_data <- data.frame(
    Cell_Status = positive_cells,
    Group = groups
  )
  
  # Perform pairwise Fisher's exact tests
  cat("Performing", length(comparisons), "pairwise comparisons...\n")
  
  pairwise_results <- lapply(comparisons, function(pair) {
    # Extract data for each group
    group1_data <- stat_data$Cell_Status[stat_data$Group == pair[1]]
    group2_data <- stat_data$Cell_Status[stat_data$Group == pair[2]]
    
    # Skip if either group has no data
    if (length(group1_data) == 0 || length(group2_data) == 0) {
      return(data.frame(
        group1 = pair[1],
        group2 = pair[2],
        p_value = NA
      ))
    }
    
    # Create contingency table
    count_table <- matrix(
      c(
        sum(group1_data), length(group1_data) - sum(group1_data),
        sum(group2_data), length(group2_data) - sum(group2_data)
      ),
      nrow = 2,
      dimnames = list(c("Positive", "Negative"), c(pair[1], pair[2]))
    )
    
    # Perform Fisher's exact test
    test_result <- tryCatch({
      fisher.test(count_table)
    }, error = function(e) {
      warning("Fisher's test failed for ", pair[1], " vs ", pair[2], ": ", e$message)
      list(p.value = NA)
    })
    
    return(data.frame(
      group1 = pair[1],
      group2 = pair[2],
      p_value = test_result$p.value
    ))
  })
  
  # Combine pairwise results
  pairwise_results <- do.call(rbind, pairwise_results)
  
  # Adjust p-values using Bonferroni correction
  valid_p <- !is.na(pairwise_results$p_value)
  pairwise_results$p_adjusted <- NA
  
  if (sum(valid_p) > 0) {
    pairwise_results$p_adjusted[valid_p] <- p.adjust(
      pairwise_results$p_value[valid_p], 
      method = "bonferroni"
    )
  }
  
  # Add significance indicators
  pairwise_results$significance <- ""
  pairwise_results$significance[pairwise_results$p_adjusted < 0.05] <- "*"
  pairwise_results$significance[pairwise_results$p_adjusted < 0.01] <- "**"
  pairwise_results$significance[pairwise_results$p_adjusted < 0.001] <- "***"
  
  # Create detailed summary statistics
  detailed_summary <- data.frame(
    Group = names(proportions),
    Positive_Count = as.vector(positive_counts),
    Total_Count = as.vector(total_counts),
    Proportion = as.vector(proportions),
    Percentage = sprintf("%.2f%%", 100 * as.vector(proportions))
  )
  
  # Apply ordering to detailed summary
  if (!is.null(desired_order)) {
    valid_order <- desired_order[desired_order %in% detailed_summary$Group]
    if (length(valid_order) > 0) {
      detailed_summary <- detailed_summary[match(valid_order, detailed_summary$Group), ]
      detailed_summary <- detailed_summary[!is.na(detailed_summary$Group), ]
    }
  }
  
  # Create visualization
  cat("Creating plot...\n")
  
  plot_title <- paste("Proportion of", output_prefix, "Cells by Treatment Group")
  
  p <- ggplot(summary_df, aes(x = Group, y = Proportion, fill = Group)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_text(aes(label = sprintf("%.3f", Proportion)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 4) +
    theme_minimal() +
    labs(
      title = plot_title, 
      x = "Treatment Group", 
      y = paste("Proportion of", output_prefix, "Cells")
    ) +
    theme(
      text = element_text(size = 12), 
      plot.title = element_text(hjust = 0.5, face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      legend.position = "none"
    )
  
  # Apply ordering to plot
  if (!is.null(desired_order)) {
    valid_order <- desired_order[desired_order %in% summary_df$Group]
    if (length(valid_order) > 0) {
      p <- p + scale_x_discrete(limits = valid_order)
    }
  }
  
  # Add significance annotations if available
  valid_comparisons <- pairwise_results[
    !is.na(pairwise_results$p_adjusted) & pairwise_results$p_adjusted < 0.05, 
  ]
  
  if (nrow(valid_comparisons) > 0) {
    if (requireNamespace("ggsignif", quietly = TRUE)) {
      
      # Create significance annotations
      annotations <- ifelse(valid_comparisons$p_adjusted < 0.001, "***",
                            ifelse(valid_comparisons$p_adjusted < 0.01, "**", "*"))
      
      # Create comparison list
      comparison_list <- lapply(1:nrow(valid_comparisons), function(i) {
        c(valid_comparisons$group1[i], valid_comparisons$group2[i])
      })
      
      # Calculate y positions for significance bars
      max_prop <- max(summary_df$Proportion, na.rm = TRUE)
      y_positions <- seq(
        from = max_prop * 1.1,
        to = max_prop * 1.1 + (0.05 * nrow(valid_comparisons)),
        length.out = nrow(valid_comparisons)
      )
      
      # Add significance brackets
      p <- p + ggsignif::geom_signif(
        comparisons = comparison_list,
        annotations = annotations,
        y_position = y_positions,
        tip_length = 0.01,
        textsize = 3
      )
      
    } else {
      message("Install 'ggsignif' package to display significance brackets on plot")
    }
  }
  
  # Save results to files
  if (save_results) {
    cat("Saving results...\n")
    
    # Save summary statistics
    write.csv(detailed_summary, 
              file = paste0(output_prefix, "_summary_stats.csv"), 
              row.names = FALSE)
    
    # Save statistical test results
    write.csv(pairwise_results, 
              file = paste0(output_prefix, "_statistical_tests.csv"), 
              row.names = FALSE)
    
    # Save plots
    ggsave(paste0(output_prefix, "_plot.pdf"), 
           plot = p, width = 10, height = 7)
    ggsave(paste0(output_prefix, "_plot.png"), 
           plot = p, width = 10, height = 7, dpi = 300)
    
    cat("Results saved with prefix:", output_prefix, "\n")
  }
  
  # Return comprehensive results
  return(list(
    seurat_object = seurat_obj,
    summary = summary_df,
    detailed_summary = detailed_summary,
    statistical_tests = pairwise_results,
    plot = p
  ))
}