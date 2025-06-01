# Load required libraries
library(ggplot2)
library(reshape2)
library(RColorBrewer)

# Create a data frame with the provided metrics
data <- data.frame(
  Domain = rep(c("Poetry", "Mental Health"), each = 5),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1-Score", "AUROC"), 2),
  Value = c(
    0.8004, 0.6714, 1.0000, 0.8034, 0.9858,  # Poetry
    0.7259, 0.6538, 0.9601, 0.7779, 0.8670   # Mental Health
  )
)

# Pivot the data for heatmap
data_pivot <- dcast(data, Domain ~ Metric, value.var = "Value")

# Convert to long format for ggplot2
data_melt <- melt(data_pivot, id.vars = "Domain", variable.name = "Metric", value.name = "Value")

# Create the heatmap
p <- ggplot(data_melt, aes(x = Metric, y = Domain, fill = Value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.4f", Value)), size = 5, fontface = "bold", color = "black") +
  scale_fill_distiller(palette = "RdBu", direction = -1, limits = c(0, 1), name = "Metric Value") +
  labs(title = "English OOD Performance Metrics Across Domains",
       x = "Metric", y = "Domain") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    legend.title = element_text(size = 10),
    legend.position = "right",
    panel.grid = element_blank(),
    panel.border = element_rect(fill = NA, color = "black", linewidth = 1)
  )

# Save the plot
ggsave("english_ood_performance_metrics.png", plot = p, width = 8, height = 5, dpi = 300)

# Display the plot
print(p)
