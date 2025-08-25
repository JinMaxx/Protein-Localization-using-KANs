library(ggplot2)
library(data.table)
library(stringr)


# Execute from Project Root!

################################################
# 0. Configuration
################################################

metrics_file <- "./data/logging/Rostlab/prot_t5_xl_uniref50/hyper_param_metrics_filtered.tsv"

output_dir <- "./data/preliminary_analysis/Rostlab/prot_t5_xl_uniref50"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

ffn_suffixes <- c("FastKAN", "MLP")  # suffic not in list will be Other

architectures_comparrison <- c("MaxPool", "AvgPool", "Attention", "UNet", "Linear", "Positional")

scatterplot_facet_order <- c(
  "MaxPool", "AvgPool", "Linear", "Attention", "Positional", "UNet",
  "AttentionLstmHybrid", "LstmReductionHybrid", "LightAttention"
)

metrics_to_summarize <- c(
  "performance", "sensitivity", "specificity", "precision",
  "accuracy", "f1_score", "roc_auc", "pr_auc", "memory_size"
)

metric_titles <- list(
  performance = "Weighted Performance Score",
  accuracy = "Accuracy",
  f1_score = "F1-Score (Macro)",
  sensitivity = "Sensitivity (Recall)",
  specificity = "Specificity",
  precision = "Precision",
  roc_auc = "ROC AUC",
  pr_auc = "PR AUC"
)

columns_to_keep_for_best_table <- c(
  "model_class", "performance", "accuracy", "f1_score", "sensitivity",
  "specificity", "precision", "roc_auc", "pr_auc", "memory_size",
  "epoch", "trial_number", "trial_params", "model"
)

columns_to_keep_for_sorted_trials_table <- c("model", "memory_size", "epoch", "trial_params")

metrics_for_sorted_trial_tables <- c("accuracy")


################################################
# 1. Data Loading and Wrangling
################################################

# Load the data, ensuring the header is read correctly
tuning_data <- fread(metrics_file, header = TRUE)

# Filter for the summary rows which contain the overall model performance
average_performance <- tuning_data[class_name == "Average"]

average_performance <- average_performance[epoch > 6]
# filter out models that converged very early (sign of possible overfitting)

# Create columns for base architecture and FFN type
suffix_pattern <- paste(ffn_suffixes, collapse = "|")

# Extract Base Architecture and FFN type
average_performance[, ffn_type := str_extract(model_class, suffix_pattern)]
average_performance[, base_architecture := str_remove(model_class, suffix_pattern)]
average_performance[is.na(ffn_type), ffn_type := "Other"]

head(average_performance)


################################################
# 2. Data Aggregation
################################################

# Group by architecture (model_class) and calculate summary statistics.
architecture_summary <- average_performance[, {
  mean_metrics <- lapply(.SD, mean, na.rm = TRUE)
  sd_metrics <- lapply(.SD, sd, na.rm = TRUE)
  names(mean_metrics) <- paste0("mean_", names(mean_metrics))
  names(sd_metrics) <- paste0("sd_", names(sd_metrics))
  c(mean_metrics, sd_metrics, list(trial_count = .N))
}, .SDcols = metrics_to_summarize, by = .(model_class, base_architecture, ffn_type)]

# If a model architecture only has one trial, its sd will be NA. We replace it with 0.
sd_cols <- names(architecture_summary)[startsWith(names(architecture_summary), "sd_")]
for (col in sd_cols) { architecture_summary[is.na(get(col)), (col) := 0] }

"Aggregated Architecture Summary:"
head(architecture_summary)


################################################
# 3. Plotting Functions
################################################

barchart <- function(summary_data, metric_name) {
  # Bars are grouped by base architecture, with transparency indicating FFN type

  metric_col <- paste0("mean_", metric_name)
  sd_col <- paste0("sd_", metric_name)

  performance_title <- metric_titles[[metric_name]]

  # Calculate the mean performance for each base_architecture group and get an ordered list of the group names.
  group_order <- summary_data[, .(group_perf = mean(.SD[[1]], na.rm = TRUE)), .SDcols = metric_col, by = base_architecture][order(-group_perf), base_architecture]

  plot_data <- summary_data

  # Set the 'base_architecture' column as a factor with the new performance-based order.
  # This allows us to sort by group performance instead of alphabetically.
  plot_data$base_architecture <- factor(plot_data$base_architecture, levels = group_order)

  # Sort the entire table: first by the new group order, then by ffn_type.
  plot_data <- plot_data[order(base_architecture, ffn_type)]

  # Set the model_class factor levels.
  plot_data$model_class <- factor(plot_data$model_class, levels = plot_data$model_class)

  # plot_data <- summary_data[order(base_architecture, ffn_type)]
  # plot_data$model_class <- factor(plot_data$model_class, levels = plot_data$model_class)

  plot <- ggplot(plot_data, aes(x = model_class, y = .data[[metric_col]], fill = base_architecture, alpha = ffn_type)) +
    geom_bar(
      stat = "identity",
      position = "dodge"
    ) +
    geom_errorbar(
      aes(ymin = .data[[metric_col]] - .data[[sd_col]],
          ymax = .data[[metric_col]] + .data[[sd_col]]),
      width = 0.4,
      position = position_dodge(0.9),
      color = "gray20"
    ) +
    geom_text(
      aes(label = paste0("N=", trial_count)),
      vjust = -2.5,
      size = 3.0,
      position = position_dodge(0.9)
    ) +
    scale_alpha_manual(
      values = c(
        "MLP" = 1.0,
        "FastKAN" = 0.65,
        "Other" = 1.0
      )
    ) +
    scale_y_continuous(
      expand = expansion(mult = c(0, 0.2)),
      breaks = seq(0, 1, by = 0.2),
      minor_breaks = seq(0, 1, by = 0.1)
    )+
    labs(
      title = paste("Average Comparison of Architectures by", performance_title),
      x = "Model Architecture",
      y = performance_title,
      fill = "Base Architecture",
      alpha = "FFN Type"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "none",
      panel.grid.major.x = element_blank(),
      panel.grid.minor.x = element_blank(),
      axis.text.x = element_text(  # angled architecture names.
        angle = 45,
        hjust = 1,
        vjust = 1,
        size = 11
      )
    )

  return(plot)
}


# Metric vs. Memory
scatterplot <- function(trial_data, metric_name) {
  # Each point is one trial. Dashed lines show per-architecture quadratic trends.

  y_axis_label <- metric_titles[[metric_name]]

  # Define shared aesthetics globally
  plot <- ggplot(trial_data, aes(x = memory_size, y = .data[[metric_name]])) +
    geom_point(
      aes(color = ffn_type),
      size = 1.5,
      stroke = 1.0,
      alpha = 0.7
    ) +
    # geom_smooth(
    #   aes(color = base_architecture, group = base_architecture),
    #   method = "lm",
    #   formula = y ~ poly(x, 2), # Fit a 2nd degree polynomial
    #   se = TRUE,
    #   linetype = "dashed",
    #   linewidth = 0.8
    # ) +
    facet_wrap(
      ~ base_architecture,
      scales = "free"
    ) +
    scale_color_manual(
      values = c(
        "MLP" = "steelblue",
        "FastKAN" = "darkorange",
        "Other" = "darkred"    # "coral"
      ),
      name = "FFN Type"
    ) +
    # scale_y_continuous(
    #   expand = expansion(mult = c(0, 0.2)),
    #   breaks = seq(0, 1, by = 0.2),
    #   minor_breaks = seq(0, 1, by = 0.1)
    # )+
    labs(
      title = paste(y_axis_label, "vs. Memory Trade-off"),
      subtitle = "",
      x = "Model Memory Size (MB)",
      y = y_axis_label
    ) +
    theme_minimal(base_size = 12) +
    theme(
      legend.position = "right",
      legend.box = "horizontal",
      strip.text = element_text(face = "bold", size = 11)
    )

  return(plot)
}


################################################
# 4. Helper Functions
################################################

save_plot <- function(plot_object, file_identifier, width = 24, height = 16, dpi = 600) {
  filename <- file.path(output_dir, paste0(file_identifier, ".png"))
  png(
    filename,
    width = width,
    height = height,
    units = "cm",
    res = dpi
  )
  print(plot_object)
  dev.off()
  cat(paste("Saved plot to:", filename, "\n"))
}


get_best_trials_by_metric <- function(trial_data, metric_name) {
  best_trials <- trial_data[, .SD[which.max(get(metric_name))], by = model_class]
  cols_that_exist <- intersect(columns_to_keep_for_best_table, names(best_trials))
  final_table <- best_trials[, ..cols_that_exist]
  final_table <- final_table[order(-get(metric_name))]
  return(final_table)
}


sort_trials_per_architecture <- function(trial_data, architecture_name, metric_name) {
  architecture_trials <- trial_data[model_class == architecture_name]
  cols_to_keep <- c(columns_to_keep_for_sorted_trials_table, metric_name)
  cols_that_exist <- intersect(cols_to_keep, names(architecture_trials))
  final_table <- architecture_trials[, ..cols_that_exist]
  final_table <- final_table[order(-get(metric_name))]
  return(final_table)
}



save_table <- function(table_object, file_identifier) {
  filename <- file.path(output_dir, paste0(file_identifier, ".tsv"))
  write.table(
    table_object,
    file = filename,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE
  )
  cat(paste("Saved table to:", filename, "\n"))
}


################################################
# 5. Plotting and Table Generation
################################################

barchart_summary_data <- architecture_summary[base_architecture %in% architectures_comparrison]

scatterplot_trial_data <- average_performance
scatterplot_trial_data$base_architecture <- factor(scatterplot_trial_data$base_architecture, levels = scatterplot_facet_order)


print("\nGenerating plots for custom 'performance' metric\n")
save_plot(
  barchart(barchart_summary_data, "performance"),
  "performance_comparrison_barchart"
)
save_plot(
  scatterplot(scatterplot_trial_data, "performance"),
  "performance_vs_memory_scatterplot"
)
save_table(
  get_best_trials_by_metric(average_performance, "performance"),
  "best_performance_table"
)


print("\nGenerating plots for 'accuracy' metric\n")
save_plot(
  barchart(barchart_summary_data, "accuracy"),
  "accuracy_comparrison_barchart"
)
save_plot(
  scatterplot(scatterplot_trial_data, "accuracy"),
  "accuracy_vs_memory_scatterplot"
)
save_table(
  get_best_trials_by_metric(average_performance, "accuracy"),
  "best_accuracy_table"
)


print("\nGenerating plots for 'F1-Score' metric\n")
save_plot(
  barchart(barchart_summary_data, "f1_score"),
  "f1_score_comparrison_barchart"
)
save_plot(
  scatterplot(scatterplot_trial_data, "f1_score"),
  "f1_score_vs_memory_scatterplot"
)
save_table(
  get_best_trials_by_metric(average_performance, "f1_score"),
  "best_f1_score_table"
)



for (metric in metrics_for_sorted_trial_tables) {
  for (arch_name in unique(average_performance$model_class)) {
    save_table(
      table_object = sort_trials_per_architecture(
        trial_data = average_performance,
        architecture_name = arch_name,
        metric_name = metric
      ),
      file_identifier = paste0("sorted_trials_for_", arch_name, "_by_", metric)
    )
  }
}