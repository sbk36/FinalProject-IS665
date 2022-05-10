# Load Libraries --------------------
library(tidyverse)  # Data wrangling and visualization
library(rpart)      # Regression trees
library(pROC)       # Receiver-operator curves

set.seed(1)

# Load and describe data --------------------
# Read in CSV file
diabetes <- read_csv("diabetes.csv")

# Summarize data
summary(diabetes)
str(diabetes)

# Spilt into training and testing
training_rows <- sample(1:nrow(diabetes), nrow(diabetes)*0.8)
training <- diabetes[training_rows,]
testing <- diabetes[-training_rows,]

# Model training ------------
mod <- rpart(Outcome ~ ., data = training, method = "class")

# Plot decision tree
png("decision-tree.png")
plot(mod)
text(mod, digits = 3)
dev.off()

# View model summary
print(mod, digits = 2)
summary(mod)

# Evaluate with testing data ------------
# Add predictions to testing dataset
testing$predicted <- predict(mod, testing, type = "class")

# Accuracy
accuracy <- mean(testing$Outcome == testing$predicted)

# Confusion matrix (CM)
# Total 
cm <- table(testing$Outcome, testing$predicted, dnn = c("Obs.", "Pred."))
cm
# CM as Probabilities
cm/nrow(testing)

# Precision and Recall
tp <- cm[2,2]
fp <- cm[1,2]
tn <- cm[1,1]
fn <- cm[2,1]

precision <- tp / (tp + fp)
recall <- tp / (tp + fn)

f1 <- 2*(recall * precision) / (recall + precision)

# ROC curve  -----------------------
# Get prediction probabilities
predict_prob <- predict(mod, testing, type = "prob")[,2]

# Calculate ROC curve using pROC package
mod_roc <- roc(testing$Outcome, predict_prob)

# Plot with ggplot
ggroc(mod_roc) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), linetype="dashed") +
  annotate("text",label = paste("AUC:",round(auc(mod_roc), 3)), x = 0.7, y=0.5) +
  labs(title = "Decision tree model performance",
       y = "True positive rate",
       x = "False positive rate") +
  theme_minimal()
ggsave("ROC-plot.png", height = 5, width = 5)

# Area under curve
auc(mod_roc)
