#### USL Clustering Project ####
#### Abdukholik Tukhtamishev ####

# The wine dataset contains different types of wines, 
# each characterized by various chemical and sensory attributes such
# as alcohol content, phenols, and color intensity. The goal of this 
# project is to identify natural groupings or clusters of wines based 
# on key factors that influence consumer preferences, such as flavor 
# profile, strength, and appearance. By performing clustering using 
#important features, we aim to uncover distinct wine categories that 
#could align with customer taste preferences or market segments, enabling 
#a better understanding of the dataset structure.

# Packages 
library(cluster)
library(gridExtra)
library(corrplot)
library(factoextra)
library(tidyverse)
library(dbscan)
library(ggplot2)
# Working directory
setwd("E:/UWarsaw/unsupervised_learning/USL_Project/Clustering")

# Import dataset
wine <- read.csv("E:/UWarsaw/unsupervised_learning/USL_Project/Clustering/wine-clustering.csv")

# General observation 
colSums(is.na(wine)) # Checking for empty cells 
head(wine)
summary(wine)
str(wine)
colnames(wine)
dim(wine)
# Ensuring data type is numeric on the grounds that k-means clustering requires 
wine <- as.data.frame(lapply(wine, function(x) as.numeric(as.character(x))))
str(wine)
wine_scale <- scale(wine)
# k-means. I decided to cluster based on top important factors and also factors that have 
# higher variance and correlation to have more vivid and cohesive clusters. First variance then 
# k-means is applied.

# 1. Calculate variance of each feature
variances <- apply(wine_scale, 2, var)

# 2. Select top 5 features with the highest variance
top_features <- names(sort(variances, decreasing = TRUE))[1:5]

# 3. Compute correlations for these top features
correlation_matrix <- cor(wine_scale[, top_features])

# 4. Plot the correlation matrix
corrplot(correlation_matrix, method = "color")

# 5. Select the least correlated features for clustering (avoid redundancy)
selected_features <- wine_scale[, c("Proline", "Total_Phenols", "Color_Intensity")]

# 6. Finding the value of k with Elbow method
set.seed(100)
wss = function(k) {kmeans(selected_features, k, nstart=25)$tot.withinss}
k.values = 1:15
wss_values = map_dbl(k.values, wss)
par(mar = c(4, 4, 2, 1))
plot(k.values, wss_values, type="b", pch=19, frame=FALSE,
     main = "Elbow plot",
     xlab = "No of clusters",
     ylab = "Total wss of squares")

# Silhouette Score
silhouette_calc <- function(selected_features, max_clusters = 10) {
  silhouette_score <- numeric(max_clusters - 1)  # Pre-allocate vector
  
  for (k in 2:max_clusters) {
    kmeans_out <- kmeans(selected_features, centers = k, nstart = 25)
    sil <- silhouette(kmeans_out$cluster, dist(selected_features))
    avg_sil <- mean(sil[, 3])  # Average silhouette width
    silhouette_score[k - 1] <- avg_sil
    cat("Silhouette Score for k =", k, ":", avg_sil, "\n")
  }
  
  # Plot Silhouette scores
  plot(2:max_clusters, silhouette_score, type = "b", pch = 19, col = "lightblue3",
       xlab = "Number of Clusters", ylab = "Average Silhouette Score",
       main = "Silhouette Method")
}

# Example usage with only numeric columns
silhouette_calc(wine_scale[, c("Proline", "Total_Phenols", "Color_Intensity")])


# Gap statistic calculation
set.seed(123)  # For reproducibility
gap_stat <- clusGap(wine_scale[, c("Proline", "Total_Phenols", 
                                   "Color_Intensity")], FUN = kmeans, nstart = 25, K.max = 10, B = 50)

# Plot the Gap Statistic
fviz_gap_stat(gap_stat) +
  labs(title = "Gap Statistic for Optimal Clusters",
       x = "Number of Clusters (k)",
       y = "Gap Statistic") +
  theme_minimal()


# Perform clustering with the selected 3 features. From the elbow point, it is 
# obvious that 3 is at the bending point
kmeans_result <- kmeans(selected_features, centers = 3, nstart = 25)

# Plot clusters
fviz_cluster(kmeans_result, data = selected_features)

# Silhouette score (3)
kmeans_result <- kmeans(selected_features, centers = 2, nstart = 25)

# Plot clusters
fviz_cluster(kmeans_result, data = selected_features)

# Gap Statistics (4)
kmeans_result <- kmeans(selected_features, centers = 4, nstart = 25)

# Plot clusters
fviz_cluster(kmeans_result, data = selected_features)


# Mean values of each cluster (Elbow Method)
means <- aggregate(selected_features, by = list
                   (cluster = kmeans_result$cluster), FUN = mean)
print(round(means, 2))


# Histogram Density Curves
# Histogram for Proline
ggplot(wine, aes(x = Proline)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "skyblue", color = "black", alpha = 0.7) +
  geom_density(color = "blue", size = 1) +
  ggtitle("Distribution of Proline") +
  xlab("Proline") +
  ylab("Density") +
  theme_minimal()

# Histogram for Total_Phenols
ggplot(wine, aes(x = Total_Phenols)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "khaki", color = "black", alpha = 0.7) +
  geom_density(color = "darkgrey", size = 1) +
  ggtitle("Distribution of Total Phenols") +
  xlab("Total Phenols") +
  ylab("Density") +
  theme_minimal()

# Histogram for Color_Intensity
ggplot(wine, aes(x = Color_Intensity)) +
  geom_histogram(aes(y = ..density..), bins = 20, fill = "coral", color = "black", alpha = 0.7) +
  geom_density(color = "darkgreen", size = 1) +
  ggtitle("Distribution of Color Intensity") +
  xlab("Color Intensity") +
  ylab("Density") +
  theme_minimal()

# Boxplots
# Boxplot for Proline
b1 <- ggplot(wine, aes(x = "", y = Proline)) +
  geom_boxplot(fill = "skyblue", color = "black") +
  ggtitle("Boxplot of Proline") +
  xlab("") +
  ylab("Proline") +
  theme_minimal()

# Boxplot for Total_Phenols
b2 <- ggplot(wine, aes(x = "", y = Total_Phenols)) +
  geom_boxplot(fill = "khaki", color = "black") +
  ggtitle("Boxplot of Total Phenols") +
  xlab("") +
  ylab("Total Phenols") +
  theme_minimal()

# Boxplot for Color_Intensity
b3 <- ggplot(wine, aes(x = "", y = Color_Intensity)) +
  geom_boxplot(fill = "coral", color = "black") +
  ggtitle("Boxplot of Color Intensity") +
  xlab("") +
  ylab("Color Intensity") +
  theme_minimal()

# Arrange plots in a single row
grid.arrange(b1, b2, b3, ncol = 3)

#### Final Conclusion ####
# The final clustering result reveals three distinct wine categories based on Proline,
# Total Phenols, and Color Intensity. Cluster 1 represents lighter wines with lower 
# Proline and phenol content, indicating a smoother profile with lower intensity.
# Cluster 2 corresponds to more balanced wines, with higher Proline and moderate phenol
# levels, suggesting medium-bodied wines. Cluster 3 captures rich, full-bodied wines 
# with high Proline and the deepest color intensity, likely appealing to those who prefer
# bold flavors and a robust structure.

# Literature review on Clustering
# Gareth James
# Daniela Witten
# Trevor Hastie
# Robert Tibshirani
#An Introduction to Statistical Learning with Applications in R





