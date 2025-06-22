

# About Dataset
# About project

#### Dimension Reduction Project ####

# Packages 
library(tidyverse)
library(ggplot2)
library(stats)
library(plotly)
library(readxl)
library(dplyr)
library(factoextra)
library(Rtsne)
library(vegan)
library(smacof)
# Working directory
setwd("E:/UWarsaw/unsupervised_learning/USL_Project/Dimension_reduction")

# Upload Pistachio data
pistachio <- read_excel("E:/UWarsaw/unsupervised_learning/USL_Project/Dimension_reduction/Pistachio_28_Features_Dataset.xlsx")

# Review Dataset
head(pistachio)
str(pistachio)
colSums(is.na(pistachio))
dim(pistachio)

# Scaling 
pistachio_scaled <- scale(select(pistachio, -Class))

# Performing PCA
pca_out <- prcomp(pistachio_scaled, center = TRUE, scale. = FALSE)
summary(pca_out)

# Choosing optimal number of PCs
fviz_eig(pca_out, addlabels = TRUE, ylim = c(0, 100)) +
  labs(title = "Scree Plot: Explained Variance by Dimensions")

# From the scree plot and cumulative proportion, we can see that 6 dimension captures about 90% of 
# variance which is accurate enough to further continue our analysis. However, we can't plot 6 dimension.
# Therefore,, for only visualization purposes, I decided  to work with 3 dimension which captures about 67%
# variance. Generally, 67% is also acceptable with high dimensional datasets

# 2D PCA Scatter Plot (PC1 vs PC2)
fviz_pca_ind(pca_out, label = "none", addEllipses = TRUE) +
  labs(title = "2D PCA Plot: PC1 vs PC2")

# 3D PCA Scatter Plot (PC1, PC2, PC3)
# Extracting PCA scores (principal component values for each sample)
pca_scores <- as.data.frame(pca_out$x)

# Create interactive 3D scatter plot
plot_ly(
  data = pca_scores,
  x = ~PC1,
  y = ~PC2,
  z = ~PC3,
  type = "scatter3d",
  mode = "markers",
  marker = list(size = 3, color = ~PC1, colorscale = "Viridis")
) %>%
  layout(
    title = list(
      text = "3D PCA Plot: PC1, PC2, and PC3",
      x = 0.5,
      y = 0.9
    ),
    scene = list(
      xaxis = list(title = "PC1-31,9%"),
      yaxis = list(title = "PC2-19,5%"),
      zaxis = list(title = "PC3-15,3%")
    )
  ) 

# Let's see how much each variable contributed to each PCA
pca_out$rotation
fviz_pca_var(pca_out, col.var = "contrib")
# The first principal component (Dim1) primarily captures color-related variability 
# (skewness and standard deviation in RGB channels), while the second component 
# (Dim2) captures geometric features (like perimeter and shape factors). 
# The longer arrows indicate that these features significantly contribute 
# to the variability explained by the first two components, while features near 
# the origin (like Eccentricity) contribute very little to these components.

# View loadings (contributions of features to each principal component)
pca_loadings <- pca_out$rotation  # Loadings matrix
# Bar plot for PC6 loadings
barplot(sort(pca_loadings[, 6], decreasing = TRUE), las = 2, col = "khaki",
        main = "Feature Contributions to PC6", ylab = "Contribution", cex.names = 0.7)

# PC6 primarily captures information related to color variability in the dataset,
# with minimal influence from shape-based features. Depending on the variance 
# explained by PC6, this component may help differentiate samples based on 
# subtle color characteristics rather than geometry. Let me know if you'd like 
# to generate more visualizations or dive deeper!


# Since PCA is linear, other methods such as t-SNE and MDS can be applied
# to see if non-linear methods provide more clear separation between 
# the pistachio classes 
# Perform t-SNE (2D)
set.seed(123)  # For reproducibility
tsne_out <- Rtsne(pistachio_scaled, dims = 2, perplexity = 30, verbose = TRUE, max_iter = 1000)

# Convert t-SNE outpt to dataframe
tsne_df <- data.frame(tsne_out$Y) 
tsne_df$Class <- pistachio$Class  
colnames(tsne_df) <- c("Dim1", "Dim2", "Class")  


# Plot t-SNE result
ggplot(tsne_df, aes(x = Dim1, y = Dim2, color = Class)) +
  geom_point(size = 2) +
  labs(title = "t-SNE Plot", x = "t-SNE Dim 1", y = "t-SNE Dim 2") +
  theme_minimal()

# From tsne plot we can see that there is part with overlap. However there are sides
# that mainly occupied by blue alone and red alone which means there is some sort of
# non-linear relationship between two types of pistachios


# Classical MDS
# Compute distance matrix
dist_matrix <- dist(pistachio_scaled)

# Classical MDS (cMDS) for 2D projection
cmds_out <- cmdscale(dist_matrix, k = 2)  # k = number of dimensions (2 for 2D)

# Dataframe with MDS results
mds_df <- data.frame(cmds_out)
mds_df$Class <- pistachio$Class  # Add class labels
colnames(mds_df) <- c("Dim1", "Dim2", "Class")

# Plot Classical MDS result
ggplot(mds_df, aes(x = Dim1, y = Dim2, color = Class)) +
  geom_point(size = 2) +
  labs(title = "Classical MDS Plot", x = "MDS Dim 1", y = "MDS Dim 2") +
  theme_minimal()

# There are several methods of MDS test. One of the well known ones is Classical MDS
# which is used when there is no linear relationship. Also it is faster than other 
# MDS methods which gives the ability to work faster with large dataset.
# If still overlapping is being observed the we can proceed with other methods of MDS

# Metric MDS
# Perform Metric MDS (2D)
set.seed(123)  # For reproducibility
metric_mds_out <- smacofSym(dist_matrix, ndim = 2)  # 'ndim = 2' for 2D projection

# Create a dataframe with MDS results
metric_mds_df <- data.frame(metric_mds_out$conf)  # 'conf' contains the 2D coordinates
metric_mds_df$Class <- pistachio$Class  # Add the original class labels
colnames(metric_mds_df) <- c("Dim1", "Dim2", "Class")

# Plot Metric MDS result
ggplot(metric_mds_df, aes(x = Dim1, y = Dim2, color = Class)) +
  geom_point(size = 2) +
  labs(title = "Metric MDS Plot", x = "MDS Dim 1", y = "MDS Dim 2") +
  theme_minimal()


# Non-Metric MDS
# Compute the distance matrix for scaled data
dist_matrix <- dist(pistachio_scaled)

# Perform Non-Metric MDS (2D)
set.seed(123)  # For reproducibility
nonmetric_mds_out <- smacofSym(dist_matrix, ndim = 2, type = "ordinal")  # "ordinal" for non-metric MDS

# Create a dataframe with MDS results
nonmetric_mds_df <- data.frame(nonmetric_mds_out$conf)
nonmetric_mds_df$Class <- pistachio$Class  # Add the original class labels
colnames(nonmetric_mds_df) <- c("Dim1", "Dim2", "Class")

# Plot Non-Metric MDS result
ggplot(nonmetric_mds_df, aes(x = Dim1, y = Dim2, color = Class)) +
  geom_point(size = 2) +
  labs(title = "Non-Metric MDS Plot", x = "MDS Dim 1", y = "MDS Dim 2") +
  theme_minimal()

# All three MDS methods showed the similar results





