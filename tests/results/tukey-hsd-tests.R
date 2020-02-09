library(agricolae)

ecbdl_results_path = '~/git/ECBDL14-Classification/tests/results/results.csv'
ecbdl_results = read.csv(ecbdl_results_path, header=TRUE, sep=',')


# Geometric Mean ANOVA + HSD
anova_results <- aov(geometric_mean ~strategy, data=ecbdl_results)
summary(anova_results)
tukey_results = HSD.test(anova_results, "strategy", group=TRUE, alpha=0.1)
print('Geometric Mean HSD Groups')
tukey_results

# True Positive Rate ANOVA + HSD
anova_results <- aov(tpr ~strategy, data=ecbdl_results)
summary(anova_results)
tukey_results = HSD.test(anova_results, "strategy", group=TRUE, alpha=0.05)
print('TPR Mean HSD Groups')
tukey_results

# True Negative Rate ANOVA + HSD
anova_results <- aov(tnr ~strategy, data=ecbdl_results)
summary(anova_results)
tukey_results = HSD.test(anova_results, "strategy", group=TRUE, alpha=0.05)
print('TNR Mean HSD Groups')
tukey_results



without_default = subset(ecbdl_results, strategy=='theoretical' | strategy=='optimal')
boxplot(tpr ~ strategy, data=without_default, 
        outline=FALSE, las=2, 
        ylab="ROC AUC",
        xlab="Minority Class Size")


library(gplots)
warnings()
# Plot the mean of teeth length by dose groups
plotmeans(tpr~strategy, data = without_default, frame = TRUE, connect = FALSE, p=0.99, n.label = FALSE, )

