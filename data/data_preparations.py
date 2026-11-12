import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

UNITS = ["kg in a m^3", "kg in a m^3", "kg in a m^3", "kg in a m^3", "kg in a m^3",
             "kg in a m^3", "kg in a m^3", "day", "(MPa, megapascals)"]

def graphs_for_dataframe(df, path):
    fig, axes = plt.subplots(ncols=2, nrows=len(df.columns), figsize=(15, 30))
    axes[0, 0].set_title("Гистограммы", fontsize=14, pad=20)
    axes[0, 1].set_title("Диаграммы Box-and-Whisker", fontsize=14, pad=20)

    for ii, column in enumerate(df.columns):
        axes[ii, 0].hist(df[column], bins=20, edgecolor='black')
        axes[ii, 0].set_ylabel(column)
        axes[ii, 0].set_xlabel(UNITS[ii])

        axes[ii, 1].boxplot(df[column], orientation='horizontal', showmeans=True)
        axes[ii, 1].set_xlabel(UNITS[ii])
        axes[ii, 1].set_yticks([])

    plt.savefig(f'{path}/histograms+box_and_whisker.png')
    plt.close()

    hmap = sns.heatmap(df.corr(),
                    square=True,
                    annot=True,
                    cmap='GnBu'
                    )
    plt.savefig(f'{path}/correlation_matrix.png')
    plt.close()

    sns.pairplot(df)
    plt.tight_layout()
    plt.savefig(f'{path}/pairplot.png')
    plt.close()

def initialize_dataframe(xls_path, diagrams_path):
    df = pd.read_excel(xls_path)
    df = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
                            'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast Furnace Slag',
                            'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly Ash',
                            'Water  (component 4)(kg in a m^3 mixture)': 'Water',
                            'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer',
                            'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse Aggregate',
                            'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine Aggregate',
                            'Age (day)': 'Age',
                            'Concrete compressive strength(MPa, megapascals) ': 'Concrete compressive strength'})
    graphs_for_dataframe(df, diagrams_path)
    return df

def clean_dataframe(df, diagrams_path):
    unique_df = df.drop_duplicates().reset_index(drop=True)

    clean_df = unique_df
    for column in clean_df.columns:
        quant1 = clean_df[column].quantile(0.25)
        quant3 = clean_df[column].quantile(0.75)
        IQR = quant3 - quant1
        lower_bound = quant1 - 1.5 * IQR
        upper_bound = quant3 + 1.5 * IQR
        index_to_delete = []
        for ii, item in enumerate(clean_df[column]):
            if item < lower_bound or item > upper_bound:
                index_to_delete.append(ii)
        clean_df = clean_df.drop(index_to_delete).reset_index(drop=True)

    print(f"Length of original dataframe: {len(df)},\n"
          f"Length without duplicates: {len(unique_df)},\n"
          f"Length without outliers: {len(clean_df)}")
    graphs_for_dataframe(clean_df, diagrams_path)

    return clean_df

def scale_dataframe(df, diagrams_path):
    scaler = StandardScaler()
    scaled_data = StandardScaler().fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
    graphs_for_dataframe(scaled_df, diagrams_path)
    return scaled_df

def prepare_dataframe(xls_path, diagrams_path):
    df = initialize_dataframe(xls_path, f"{diagrams_path}/initial")
    clean_df = clean_dataframe(df, f"{diagrams_path}/cleanned")
    scaled_dataframe = scale_dataframe(clean_df, f"{diagrams_path}/scaled")
    return scaled_dataframe