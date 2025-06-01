# Load required libraries
library(ggplot2)
library(reshape2)
library(RColorBrewer)

# Create a data frame with the provided metrics
data <- data.frame(
  Domain = rep(c("Law", "Finance", "Medicine"), each = 5),
  Metric = rep(c("Accuracy", "Precision", "Recall", "F1-Score", "AUROC"), 3),
  Value = c(
    0.5448, 0.4814, 0.9980, 0.6495, 0.9618,  # Law
    0.6376, 0.6070, 0.9995, 0.7553, 0.9562,  # Finance
    0.6145, 0.5647, 1.0000, 0.7218, 0.9802   # Medicine
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
  scale_fill_disctiller(palette = "RdBu", direction = -1, limits = c(0, 1), name = "Metric Value") +
  labs(title = "Chinese OOD Performance Metrics Across Domains",
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
ggsave("chinese_ood_performance_metrics.png", plot = p, width = 8, height = 5, dpi = 300)

# Display the plot
print(p)
