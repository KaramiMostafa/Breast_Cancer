import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the CSV files
test_metrics = pd.read_csv('features_variation_test_senario_II.csv')
train_metrics = pd.read_csv('features_variation_train_senario_II.csv')

# Set the number of features as the index for both dataframes
test_metrics = test_metrics.set_index('features_number')
train_metrics = train_metrics.set_index('features_number')

# Create a grid plot with two subplots
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 8))

# Plot the mean accuracy for train and test sets with std as error bars
x = np.arange(len(test_metrics))
ax1.bar(x - 0.2, test_metrics['mean_accuracy_LR'], width=0.4, label='Test set', yerr=test_metrics['std_accuracy_LR'])
ax1.bar(x + 0.2, train_metrics['mean_accuracy_LR'], width=0.4, label='Train set', yerr=train_metrics['std_accuracy_LR'])
ax1.set_ylabel('Mean accuracy')
ax1.legend(loc='lower right')
ax1.set_xticks(x)
ax1.set_xticklabels(test_metrics.index)
ax1.tick_params(axis='x', labelrotation=45)

# Plot the standard deviation for train and test sets
ax2.bar(x - 0.2, test_metrics['std_accuracy_LR'], width=0.4, label='Test set')
ax2.bar(x + 0.2, train_metrics['std_accuracy_LR'], width=0.4, label='Train set')
ax2.set_xlabel('Number of features')
ax2.set_ylabel('Standard deviation')
ax2.legend(loc='upper right')
ax2.set_xticks(x)
ax2.set_xticklabels(test_metrics.index)
ax2.tick_params(axis='x', labelrotation=45)

# Add a title to the plot
plt.suptitle('Comparison of test and train accuracy results')

# Adjust the layout
plt.tight_layout()

# Save the plot as a high-resolution PNG file
plt.savefig('accuracy_std_comparison.png', dpi=300)

