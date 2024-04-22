import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects
import matplotlib.patches as mpatches
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

def box_plot_percentages_experiments(df:pd.DataFrame, rank_df:pd.DataFrame, params:dict, color_ranks:bool, unique_ranks:list=None, color_palette:str="viridis", pairwise:bool=False, ylim:float=None, yscale:str=None, figsize:tuple=None, add_baseline:bool=True):
    # Creating subplots for each data percentage
    unique_percentages = df['Percentage'].unique()
    n_percentages = len(unique_percentages)

    if color_ranks:
        if unique_ranks is None:
            unique_ranks = rank_df['rank'].unique()
        ranks_lim = unique_ranks.max()
        rank_color_map = {rank: ranks_lim+1-rank for rank in range(ranks_lim, 0, -1)}
        # print(rank_color_map)

    if figsize is None:
        figsize = (10, 2 * n_percentages)

    # Adjusting the subplot layout for better readability of median values
    fig, axes = plt.subplots(nrows=n_percentages, ncols=1, figsize=figsize, sharex=True)

    for i, percentage in enumerate(sorted(unique_percentages)):
        # Filtering data for each percentage
        df_subset = df[df['Percentage'] == percentage]

        if color_ranks:
            # Custom coloring based on ranks sorted by median accuracy
            rank_subset = rank_df[rank_df['Percentage'] == percentage]
            ranks = rank_subset.set_index('Cut Point')['rank']
            if add_baseline:
                ranks.loc[-1] = ranks.loc[0]   
            ranks_sorted = ranks.sort_values(ascending=False)

            # new_ranks = ranks_sorted.copy()
            # current_rank = ranks_sorted.iloc[-1]
            # prev = current_rank
            # for k in range(len(ranks_sorted)-2, -1, -1):
            #     if ranks_sorted.iloc[k] != prev:
            #         prev = ranks_sorted.iloc[k]
            #         current_rank += 1
            #     new_ranks.iloc[k] = current_rank
            
            cut_color_map = {cut: rank_color_map[rank] for cut, rank in ranks_sorted.items()}
            cut_rank_map = {cut: rank for cut, rank in ranks_sorted.items()}
            color_palette = sns.color_palette(color_palette, len(rank_color_map)+1)
            palette = {cut: color_palette[color_id] for cut, color_id in cut_color_map.items()}

            sns.boxplot(x='Cut Point', y='Test Accuracy', data=df_subset, ax=axes[i], palette=palette)
        else:
            # Creating a boxplot for the current percentage with default coloring
            sns.boxplot(x='Cut Point', y='Test Accuracy', data=df_subset, ax=axes[i])

        # Annotating each boxplot with the median value for better contrast
        medians = df_subset.groupby(['Cut Point'])['Test Accuracy'].median().sort_index()
        for j, median in enumerate(medians):
            txt = f'{median:.3f}\nRank: {cut_rank_map[j-1]}' if color_ranks else f'{median:.3f}'
            if pairwise:
                txt = f'{median:.3f}'
            text = axes[i].text(j, median, txt, horizontalalignment='center', size='small', color='white', weight='semibold')
            text.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])

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

    # Adding legend
    if pairwise:
        handles = [mpatches.Patch(color=color_palette[0], label='Significantly Higher'), mpatches.Patch(color=color_palette[len(color_palette)//2], label='Not Significant'), mpatches.Patch(color=color_palette[len(color_palette)-1], label='Significantly Lower')]
        fig.legend(handles=handles, loc='upper right')

    # Adding a super title
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

# ---------------------------------------------------------- STATISTICAL TESTS --------------------------------------------------------
from scipy.stats import wilcoxon

def pairwise_comparison(df:pd.DataFrame):
    df_differences = df[df['Cut Point'] != -1]

    def perform_wilcoxon_test(group, cut_point_1, cut_point_2):
        data_1 = group[group['Cut Point'] == cut_point_1]['Test Accuracy']
        data_2 = group[group['Cut Point'] == cut_point_2]['Test Accuracy']

        # Ensure equal length by trimming or padding
        min_len = min(len(data_1), len(data_2))
        data_1, data_2 = data_1[:min_len], data_2[:min_len]

        stat, p_value = wilcoxon(data_1, data_2)
        return stat, p_value

    # Perform pairwise comparison for each sampled_percentage
    wilcoxon_pairwise_results = []

    for percentage in df_differences['Percentage'].unique():
        group = df_differences[df_differences['Percentage'] == percentage]
        cut_points = group['Cut Point'].unique()

        for i in range(len(cut_points)):
            for j in range(i + 1, len(cut_points)):
                stat, p_value = perform_wilcoxon_test(group, cut_points[i], cut_points[j])
                wilcoxon_pairwise_results.append({
                    'Percentage': percentage,
                    'Cut Point 1': cut_points[i],
                    'Cut Point 2': cut_points[j],
                    'statistic': stat,
                    'p_value': p_value
                })

    # Converting the results to a DataFrame
    df_wilcoxon_pairwise = pd.DataFrame(wilcoxon_pairwise_results)
    df_wilcoxon_pairwise["is_significant"] = df_wilcoxon_pairwise["p_value"] < 0.05

    # Assuming df_wilcoxon_pairwise and df_differences are defined as before

    # Step 1: Aggregate Test Accuracy and Initialize Rankings
    median_accuracy = df_differences.groupby(['Percentage', 'Cut Point'])['Test Accuracy'].median()

    rankings = {percentage: {} for percentage in median_accuracy.index.get_level_values(0).unique()}
    for (percentage, cut_point), accuracy in median_accuracy.items():
        rankings[percentage][cut_point] = {'median_accuracy': accuracy, 'rank': None, 'group': set([cut_point])}

    # Step 2: Group Cuts Based on Statistical Significance
    alpha = 0.05  # Significance level

    for percentage, cuts_info in rankings.items():
        for cut1, info1 in cuts_info.items():
            for cut2, info2 in cuts_info.items():
                if cut1 != cut2:
                    wilcoxon_row = df_wilcoxon_pairwise[
                        (df_wilcoxon_pairwise['Percentage'] == percentage) & 
                        ((df_wilcoxon_pairwise['Cut Point 1'] == cut1) & (df_wilcoxon_pairwise['Cut Point 2'] == cut2) |
                        (df_wilcoxon_pairwise['Cut Point 1'] == cut2) & (df_wilcoxon_pairwise['Cut Point 2'] == cut1))
                    ]
                    if not wilcoxon_row.empty and wilcoxon_row.iloc[0]['p_value'] > alpha:
                        # Merge groups if not significantly different
                        info1['group'].update(info2['group'])
                        for cut in info2['group']:
                            info1['group'].update(rankings[percentage][cut]['group'])
                        for cut in info1['group']:
                            rankings[percentage][cut]['group'] = info1['group']
                    

    # Step 3: Assign Ranks with Ties
    for percentage, cuts_info in rankings.items():
        sorted_groups = sorted({frozenset(info['group']) for cut, info in cuts_info.items()}, key=lambda x: min(median_accuracy[percentage, cut] for cut in x), reverse=True)
        rank = 1
        for group in sorted_groups:
            for cut in group:
                rankings[percentage][cut]['rank'] = rank
            rank += len(group)

    # Convert rankings and median accuracies to a DataFrame for easier visualization
    df_rankings = pd.DataFrame([(percentage, cut, data['rank'], data['median_accuracy']) for percentage, cuts_info in rankings.items() for cut, data in cuts_info.items()],
                            columns=['Percentage', 'Cut Point', 'rank', 'median_accuracy'])
    
    return df_wilcoxon_pairwise, df_rankings

def perform_pairwise_wilcoxon_test(data_1, data_2, default=5):
    if np.allclose(data_1, data_2):
        return 0, 1
    # Ensure equal length by trimming or padding
    min_len = min(len(data_1), len(data_2))
    data_1, data_2 = data_1[:min_len], data_2[:min_len]

    stat, p_value = wilcoxon(data_1, data_2)
    return stat, p_value

def pairwise_comparison_multiple_plots(df1:pd.DataFrame, df2:pd.DataFrame):
    # Perform pairwise comparison for each sampled_percentage
    wilcoxon_pairwise_results = []

    for percentage in df1['Percentage'].unique():
        df1_perc = df1[df1['Percentage'] == percentage]
        df2_perc = df2[df2['Percentage'] == percentage]
        
        cut_points = df1_perc['Cut Point'].unique()

        for cut in cut_points:
            stat, p_value = perform_pairwise_wilcoxon_test(df1_perc[df1_perc['Cut Point'] == cut]['Test Accuracy'],
                                                        df2_perc[df2_perc['Cut Point'] == cut]['Test Accuracy'])
            wilcoxon_pairwise_results.append({
                'Percentage': percentage,
                'Cut Point': cut,
                'statistic': stat,
                'p_value': p_value
            })

    # Converting the results to a DataFrame
    df_wilcoxon_pairwise = pd.DataFrame(wilcoxon_pairwise_results)

    df_wilcoxon_pairwise['is_significant'] = df_wilcoxon_pairwise['p_value'] < 0.05

    return df_wilcoxon_pairwise

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
