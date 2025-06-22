# Project Association rule mining

# About data set 
# About project

# Libraries 
library(tidyverse)
library(dplyr)
library(arules)
library(arulesViz)
library(ggplot2)


# Working directory
setwd("E:/UWarsaw/unsupervised_learning/USL_Project/Association")
getwd()

# Upload dataset
basket <- read.csv("E:/UWarsaw/unsupervised_learning/USL_Project/Association/market.csv")

# General Summary
head(basket)
dim(basket)
colSums(is.na(basket))

# It turns out that data in dataset is merged into single column. We are gonna seperate them.
basket_sep <- basket %>%
  separate(products, sep = ";", into = c("Bread", "Honey",
                                         "Bacon", "Toothpaste",
                                         "Banana", "Apple",
                                         "Hazelnut", "Cheese",
                                         "Meat", "Carrot", 
                                         "Cucumber", "Onion",
                                         "Milk", "Butter",
                                         "ShavingFoam", "Salt",
                                         "Flour", "HeavyCream",
                                         "Egg", "Olive", 
                                         "Shampoo", "Sugar"))
head(basket_sep)

# Removing the row with product name 'basket_sep'
basket_sep <- basket_sep[-1, ]
head(basket_sep)

# Transactions
# Convert to transaction format

# Convert binary matrix to a list of transactions
transactions <- apply(basket_sep, 1, function(row) {
  colnames(basket_sep)[row == 1]  # Select column names where value is 1
})

# Convert the list into a transactions object
transactions <- as(transactions, "transactions")

# View summary of transactions
summary(transactions)
inspect(transactions[1:5])  # Inspect first 5 transactions

# Plot item frequency (Top 10 products)
itemFrequencyPlot(transactions, topN = 10, type = "absolute", main = "Top 10 Purchased Products")

# Apriori algorithm
rules_apriori <- apriori(transactions, parameter = list(supp = 0.02, conf = 0.8))
summary(rules_apriori)

# Inspect top 10-Rules
inspect(sort(rules_apriori, by = "lift")[1:10])

# Keep only the top 100 rules by lift
top_rules <- head(sort(rules_apriori, by = "lift"), 100)

# Plot the top-rule set
plot(top_rules, method = "scatterplot", measure = c("support", "confidence"), shading = "lift")

# Grouped matrix plot
plot(top_rules, method = "grouped")

# Graph-Based Visualization
plot(top_rules, method = "graph", engine = "htmlwidget")  # Requires a browser for interactive viewing
plot(top_rules, method = "graph", control = list(type = "items"))

# Parallel coordinates plot
plot(top_rules, method = "paracoord", control = list(reorder = TRUE))



# ECLAT

# Apply Eclat to find frequent itemsets
frequent_itemsets <- eclat(transactions, parameter = list(supp = 0.01, maxlen = 3))

# Inspect the frequent itemsets
inspect(frequent_itemsets)

# Sorting frequent itemsets
sorted_itemsets <- sort(frequent_itemsets, by = "support", decreasing = TRUE)
inspect(sorted_itemsets)

# Top 10 itemsets
top_10 <- head(sorted_itemsets)
inspect(top_10)

# Frequent itemsets with exactly 3 items
triple_itemsets <- frequent_itemsets[size(frequent_itemsets) == 3]

# Sort by support
sorted_triple_itemsets <- sort(triple_itemsets, by = "support", decreasing = TRUE)

# Select the top N triple itemsets
top_triple_itemsets <- head(sorted_triple_itemsets, 10)

# Plot a bar chart for the top triple itemsets
top_triple_df <- as(top_triple_itemsets, "data.frame")
ggplot(top_triple_df, aes(x = reorder(items, support), y = support)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Top Triple Frequent Itemsets",
    x = "Itemsets",
    y = "Support"
  ) +
  theme_minimal()

# Network plot
plot(top_triple_itemsets, method = "graph", engine = "igraph", 
     main = "Network of Top Triple Frequent Itemsets")


# Scatter plot
ggplot(top_triple_df, aes(x = items, y = support)) +
  geom_point(color = "steelblue", size = 3) +
  labs(title = "Scatter Plot of Top Triple Frequent Itemsets", x = "Itemsets", y = "Support") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels

# Matrix plot
plot(top_triple_df, method = "matrix", 
     main = "Matrix Plot of Top Triple 10 Itemsets")

# Comparing two methods
# For Apriori
length(rules_apriori)

# For Eclat
length(frequent_itemsets)

# Overlap
# Extract itemsets from Apriori
apriori_itemsets <- unique(items(rules_apriori))

# Extract itemsets from ECLAT
eclat_itemsets_list <- unique(items(frequent_itemsets))

# Compare overlap
overlap <- intersect(apriori_itemsets, eclat_itemsets_list)
length(overlap)  # Number of overlapping itemsets

# Compare performance
# Apriori timing
time_apriori <- system.time({
  rules_apriori <- apriori(transactions, 
                           parameter = list(supp = 0.01, conf = 0.6, maxlen = 8))
})
print(time_apriori)

# Eclat timing
time_eclat <- system.time({
  frequent_itemsets <- eclat(transactions, 
                          parameter = list(supp = 0.01, maxlen = 8))
})
print(time_eclat)

# Compare timing of two methods
cat("Apriori runtime: ", time_apriori["elapsed"], "seconds\n")
cat("ECLAT runtime: ", time_eclat["elapsed"], "seconds\n")


# Conclusion



