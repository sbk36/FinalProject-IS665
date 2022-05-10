# Load Libraries --------------------
library(tidyverse)  # Data wrangling and visualization
library(rpart)      # Regression trees

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
