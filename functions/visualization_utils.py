import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
import pandas as pd
import seaborn as sns
import json, copy
from scipy.interpolate import griddata

# --------------------------------------------------- EMPIRICAL EXPERIMENTS --------------------------------------------------------
def calculate_statistics(df:pd.DataFrame):
    df['Mean Train Accuracy'] = df.groupby(['learning rate', 'Percentage', 'Cut Point'])['Train Accuracy'].transform('mean')
    df['Median Train Accuracy'] = df.groupby(['learning rate', 'Percentage', 'Cut Point'])['Train Accuracy'].transform('median')
    df['Mean Test Accuracy'] = df.groupby(['learning rate', 'Percentage', 'Cut Point'])['Test Accuracy'].transform('mean')
    df['Median Test Accuracy'] = df.groupby(['learning rate', 'Percentage', 'Cut Point'])['Test Accuracy'].transform('median')
    df['Max Test Accuracy'] = df.groupby(['learning rate', 'Percentage', 'Cut Point'])['Test Accuracy'].transform('max')
    df['Max Train Accuracy'] = df.groupby(['learning rate', 'Percentage', 'Cut Point'])['Train Accuracy'].transform('max')
    return df


def box_plot_percentages_experiments(df:pd.DataFrame, params:dict, ylim:float=None, yscale:str=None, figsize:tuple=(10,20)):
    # Creating subplots for each data percentage
    unique_percentages = df['Percentage'].unique()
    #unique_percentages = [0.001, 0.002, 0.005, 0.01, 0.05, 0.1]
    n_percentages = len(unique_percentages)

    # Adjusting the subplot layout for better readability of median values (improving contrast)
    fig, axes = plt.subplots(nrows=n_percentages, ncols=1, figsize=figsize, sharex=True)

    for i, percentage in enumerate(sorted(unique_percentages)):
        # Filtering data for each percentage
        df_subset = df[df['Percentage'] == percentage]
        
        # Creating a boxplot for the current percentage
        sns.boxplot(x='Cut Point', y='Test Accuracy', data=df_subset, ax=axes[i])
        axes[i].set_title(f'Sampled Percentage: {percentage}')
        axes[i].set_xlabel('Sampled Cut Point')
        if ylim:
            axes[i].set_ylim(ylim, 1.0)
        if yscale:
            axes[i].set_yscale('log')

        if i == n_percentages - 1:
            axes[i].set_xlabel('Sampled Cut Point')
        else:
            axes[i].set_xlabel('')
        axes[i].set_ylabel('Test Accuracy')

        # Annotating each boxplot with the median value and adjusting for better contrast
        medians = df_subset.groupby(['Cut Point'])['Test Accuracy'].median().sort_index()
        for j, median in enumerate(medians):
            text = axes[i].text(j, median, f'{median:.3f}', 
                                horizontalalignment='center', size='small', color='white', weight='semibold')
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])
    st = fig.suptitle(f'Freeze = {params["freeze"]}, Reinitialize = {params["reinit"]}, Pooling = {params["use_pooling"]}, Learning rate = {params["lr_fine_tune"]}')

    st.set_y(1.0)
    fig.subplots_adjust(top=0.85)
    plt.tight_layout()
    return plt

def box_plot_full_experiments(df:pd.DataFrame, params:dict):
    # Creating boxplots
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Percentage', y='Test Accuracy', hue='Cut Point', data=df)
    plt.title(f'Candlestick Plot: Freeze = {params["freeze"]}, Reinitialize = {params["reinit"]}, Pooling = {params["use_pooling"]}, Learning rate = {params["lr_fine_tune"]}')
    plt.xlabel('Sampled Percentage')
    plt.ylabel('Test Accuracy')
    plt.legend(title='Sampled Cut Point')
    return plt

def compare_to_baseline_line_plot(df:pd.DataFrame, params:dict=None, figsize:tuple=(14, 6)):
    # Grouping by sampled_percentage, sampled_cut_point and calculating medians
    medians_by_cut = df.groupby(['Percentage', 'Cut Point'])['Test Accuracy'].median().reset_index()

    # Pivoting the table for easier calculation of differences
    pivot_medians = medians_by_cut.pivot(index='Percentage', columns='Cut Point', values='Test Accuracy')

    # Identifying all unique cut points except for the baseline (-1)
    cut_points = [col for col in pivot_medians.columns if col != -1]

    # Calculating differences from baseline for each cut point
    for cut in cut_points:
        pivot_medians[f'diff_cut_{cut}'] = pivot_medians[cut] - pivot_medians[-1]

    # Resetting index to make 'sampled_percentage' a column
    pivot_medians.reset_index(inplace=True)

    # Removing rows with NaN values in the new difference columns
    pivot_medians.dropna(subset=[f'diff_cut_{cut}' for cut in cut_points], inplace=True)

    # Plot setup
    plt.figure(figsize=figsize)
    palette = sns.color_palette("Set2", len(cut_points))  # Setting palette based on number of cut points
    plt.title('Differences in Median Test Accuracy Compared to Baseline (Cut -1)')
    plt.xlabel('Sampled Percentage (Log Scale)')
    plt.ylabel('Difference in Median Test Accuracy')
    plt.axhline(0, color='gray', linestyle='--')  # Reference line at zero

    # Plotting the differences for each cut
    for i, cut in enumerate(cut_points):
        sns.lineplot(x='Percentage', y=f'diff_cut_{cut}', data=pivot_medians, marker='o', color=palette[i])

    # Adding vertical dashed lines for each sampled percentage
    for percentage in pivot_medians['Percentage'].unique():
        plt.axvline(percentage, color='gray', linestyle='--', alpha=0.5)

    plt.xscale('log')
    plt.xticks(pivot_medians['Percentage'], labels=pivot_medians['Percentage'], rotation=45)

    # Custom legend
    legend_labels = [f'Cut {cut}' for cut in cut_points]
    plt.legend(title='Cut Point', labels=legend_labels, handles=[plt.Line2D([0], [0], color=palette[i], marker='o') for i in range(len(cut_points))])
    return plt

def line_plot(df:pd.DataFrame, params:dict, figsize:tuple=(10, 6)):
    # Calculate mean and standard deviation for test accuracies
    # Assuming that the provided data has a pattern where every three tuples belong to the same
    # percentage and cut point but different trials
    df['Mean Test Accuracy'] = df.groupby(['Percentage', 'Cut Point'])['Test Accuracy'].transform('mean')
    df['Std Test Accuracy'] = df.groupby(['Percentage', 'Cut Point'])['Test Accuracy'].transform('std')

    # Plotting
    plt.figure(figsize=figsize)

    # Iterate over each unique cut point
    for cut_point in df['Cut Point'].unique():
        cut_df = df[df['Cut Point'] == cut_point]

        # Calculate mean and standard deviation for each percentage
        means = cut_df.groupby('Percentage')['Mean Test Accuracy'].mean()
        stds = cut_df.groupby('Percentage')['Std Test Accuracy'].mean()

        # Plot with error bars for uncertainty
        plt.errorbar(means.index, means, yerr=stds, capsize=5, label=f'Cut {cut_point}')
        #plt.errorbar(means.index, means, capsize=5, label=f'Cut {cut_point}')

    plt.xlabel('Percentage')
    plt.ylabel('Test Accuracy')
    plt.title(f'Freeze = {params["freeze"]}, Reinitialize = {params["reinit"]}, Pooling = {params["use_pooling"]}, Learning rate = {params["lr_fine_tune"]}')
    plt.legend()
    # x log scale
    plt.xscale('log')
    plt.grid(True)
    return plt

def line_pilot_with_ranges(df:pd.DataFrame, params:dict, figsize:tuple=(10, 6)):
    # Calculate median, min, and max for test accuracies
    df['Median Test Accuracy'] = df.groupby(['Percentage', 'Cut Point'])['Test Accuracy'].transform('median')
    df['Min Test Accuracy'] = df.groupby(['Percentage', 'Cut Point'])['Test Accuracy'].transform('min')
    df['Max Test Accuracy'] = df.groupby(['Percentage', 'Cut Point'])['Test Accuracy'].transform('max')

    # Plotting
    plt.figure(figsize=figsize)

    # Iterate over each unique cut point
    for cut_point in df['Cut Point'].unique():
        cut_df = df[df['Cut Point'] == cut_point]

        # Calculate median, min, and max for each percentage
        medians = cut_df.groupby('Percentage')['Median Test Accuracy'].median()
        mins = cut_df.groupby('Percentage')['Min Test Accuracy'].min()
        maxs = cut_df.groupby('Percentage')['Max Test Accuracy'].max()

        # Plot median values
        plt.plot(medians.index, medians, label=f'Cut {cut_point}')

        # Plot intervals between min and max as shaded areas
        plt.fill_between(medians.index, mins, maxs, alpha=0.2)

    plt.xlabel('Percentage')
    plt.ylabel('Test Accuracy')
    plt.title(f'Freeze = {params["freeze"]}, Reinitialize = {params["reinit"]}, Pooling = {params["use_pooling"]}, Learning rate = {params["lr_fine_tune"]}')
    plt.legend()
    plt.grid(True)
    return plt

def heatmap(df:pd.DataFrame, params:dict, figsize:tuple=(10,20)):
    # For the heatmap, we will use the mean test accuracy, aggregated over trials for each combination of 'Percentage' and 'Cut Point'
    # Calculate the mean/max statistics and add them to the df
    df = calculate_statistics(df)

    # We will pivot the dataframe to get the mean test accuracy for each combination of 'Percentage' and 'Cut Point'
    heatmap_data = df.pivot_table(index='Percentage', columns='Cut Point', values='Mean Test Accuracy', aggfunc='max')

    plt.figure(figsize=figsize)
    sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt=".2f")
    plt.title(f'Mean Test Accuracy: Freeze = {params["freeze"]}, Reinitialize = {params["reinit"]}, Pooling = {params["use_pooling"]}, Learning rate = {params["lr_fine_tune"]}')
    plt.xlabel('Cut Point')
    plt.ylabel('Percentage')
    return plt

# Create a bar graph for comparing the test accuracies of different cut points at each percentage level.
def bar_graph(df:pd.DataFrame, params:dict, figsize:tuple=(10,20)):
    # Find the mean test accuracy for each cut point at each percentage level.
    mean_accuracies = df.groupby(['Cut Point', 'Percentage']).mean()['Mean Test Accuracy'].unstack(0)

    # Plotting
    mean_accuracies.plot(kind='bar', figsize=figsize, width=0.8)

    plt.title(f'Freeze = {params["freeze"]}, Reinitialize = {params["reinit"]}, Pooling = {params["use_pooling"]}, Learning rate = {params["lr_fine_tune"]}')
    plt.xlabel('Percentage')
    plt.ylabel('Mean Test Accuracy')
    plt.xticks(rotation=0)  # Rotate x-axis labels to show them horizontally.
    plt.legend(title='Cut Points')
    plt.tight_layout()
    return plt

def surface_plot(df:pd.DataFrame, params:dict=None, figsize:tuple=(12, 8)):
    # Calculate the mean/max statistics and add them to the df
    df = calculate_statistics(df)

    # Create grid values
    xi = np.linspace(df['Percentage'].min(), df['Percentage'].max(), 100)
    yi = np.linspace(df['Cut Point'].min(), df['Cut Point'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate
    zi = griddata((df['Percentage'], df['Cut Point']), df['Mean Test Accuracy'], (xi, yi), method='cubic')

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xi, yi, zi, cmap='viridis')
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Cut Point')
    ax.set_zlabel('Mean Test Accuracy')
    return plt

# ---------------------------------------------------------- KP METRICS --------------------------------------------------------
# dataset: "Finetune" or "Pretrain"
# split: "Train" or "Val" or "Test"
def plot_ARI_scores_percentages(df:pd.DataFrame, dataset:str, split:str, order=None, ylim:float=None, yscale:str=None, figsize:tuple=(10,20)):
    # Filter the data with "Dataset"=dataset and "Split"=split: because we might save different versions in one json too
    df = df[df["Dataset"] == dataset]
    df = df[df["Split"] == split]
    
    # Creating subplots for each data percentage
    unique_percentages = df['Percentage'].unique()
    n_percentages = len(unique_percentages)

    # Adjusting the subplot layout for better readability of median values (improving contrast)
    fig, axes = plt.subplots(nrows=n_percentages, ncols=1, figsize=figsize, sharex=True)

    for i, percentage in enumerate(sorted(unique_percentages)):
        # Filtering data for each percentage
        df_subset = df[df['Percentage'] == percentage]
        
        # Creating a boxplot for the current percentage
        sns.boxplot(x='Layer', y='Max ARI Score', data=df_subset, ax=axes[i], order=order)
        axes[i].set_title(f'Percentage: {percentage}, Num Samples: {df_subset.iloc[0]["Num Samples"]}')
        axes[i].set_xlabel('Layer')
        if ylim:
            axes[i].set_ylim(ylim, 1.0)
        if yscale:
            axes[i].set_yscale('log')

        if i == n_percentages - 1:
            axes[i].set_xlabel('Layer')
        else:
            axes[i].set_xlabel('')
        axes[i].set_ylabel('Max ARI Score')

        # Annotating each boxplot with the median value and adjusting for better contrast
        medians = df_subset.groupby(['Layer'])['Max ARI Score'].median().sort_index()
        for j, median in enumerate(medians):
            text = axes[i].text(j, median, f'{median:.3f}', 
                                horizontalalignment='center', size='small', color='white', weight='semibold')
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])
    st = fig.suptitle(f'ARI Scores, {dataset} {split} dataset')

    st.set_y(1.0)
    fig.subplots_adjust(top=0.85)
    plt.tight_layout()
    return plt

def plot_ARI_scores(ari_scores, channel_ids, layer_names, num_samples:str, dataset:str, split:str):
    # Extract mean and std for each layer
    mean_scores = [np.mean(list(ari_scores[layer].values())) for layer in layer_names]
    std_scores = [np.std(list(ari_scores[layer].values())) for layer in layer_names]

    # Create a box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot([list(ari_scores[layer].values()) for layer in layer_names], showmeans=True)

    plt.xticks(range(1, len(layer_names) + 1), [name+"\nmean: "+str(round(mean_scores[i]))+"\nstd: "+str(round(std_scores[i])) for i, name in enumerate(layer_names)])
    plt.xlabel('Layer')
    plt.ylabel('Scores (*100)')
    plt.title(f'ARI Scores for each CNN layer, {len(channel_ids)} channels, {num_samples} samples and {dataset} {split} dataset')
    plt.show()
