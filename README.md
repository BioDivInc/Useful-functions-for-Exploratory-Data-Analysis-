# Useful functions for Exploratory Data Analysis

I've come up with some, hopefully, usefull functions to simplify EDA of a loaded dataset. I'd recommend using an '.ipynb' file for it to work properly and display tables nicely.

## 1. Print out basic information about the loaded dataset
The first function prints basic information about the dataset (e.g., dimensions, dtypes, checks for NAs, ...) to skip the very first step of just getting a feeling for the dataset and checking the basics. The function includes, instead of basics tables, a prettier output using `tabulate` with `tablefmt='rounded_grid'`.

```python
def print_bascis(dataset):
    """
    Prints out basic information about a loaded dataset:\n
    Sample of 10 rows,\n
    Dataset dimensions,\n
    Dataset dtypes,\n
    Dataset dtype summary,\n
    NAs check,\n
    Description of numeric features.
    """
    # install and import 'tabulate' for a cleaner output of dataset as well as pandas 
    try:
        from tabulate import tabulate
        print("Imported tabulate")
        import pandas as pd
        print("Imported pandas.")
        from IPython.display import display
        print("Imported display.")
    except:
        %pip install tabulate
        from tabulate import tabulate
        print("Installed and imported tabulate.")
        %pip install pandas
        import pandas as pd
        print("Installed and imported pandas.")

    # set options to display dataset
    pd.set_option('display.width', 1000) # control width
    pd.set_option('display.max_columns', None)  # show all columns
    pd.set_option('display.max_colwidth', None)  # don't truncate column content

    # print a sample of dataset
    print(f"\nSample of 10 rows of the dataset:")
    print(tabulate(dataset.sample(10), headers='keys', tablefmt='rounded_grid'))

    # check the dimensions of the dataset
    print(f"Dataset dimensions: {dataset.shape[0]} rows and {dataset.shape[1]} columns.")

    # print out the dtype for each column found
    dtype_table = [(col, dtype) for col, dtype in dataset.dtypes.items()]
    print("\nColumn data types:")
    print(tabulate(dtype_table, headers=["column name", "data type"], tablefmt="rounded_grid"))

    # print summary table for dtypes
    df_dtype = pd.DataFrame(dtype_table, columns=['column name', 'data type'])
    print("\nSummary table dtypes:")
    print(tabulate(
        df_dtype['data type'].value_counts().reset_index().values.tolist(),
        headers=['data type', 'count'],
        tablefmt='rounded_grid'
    ))

    # check for NAs
    print("\nLooking for NAs...", end="")
    if dataset.isna().sum().sum() > 0:
        print("Found NAs:")
        df_na = dataset.isna().sum()
        print(tabulate(df_na.reset_index().values.tolist(), headers=['column name', 'count'], tablefmt='rounded_grid'))
    else:
        print("No NAs found.")
    
    # print out a description of numeric values (mean, max, ...)
    dataset = dataset.select_dtypes(include=['int', 'float'])
    print("\nSummary of numerical features:")
    display(dataset.describe().T.style.format("{:.2f}").background_gradient(cmap='Blues')) # description of numerical features with transposed rows, backed with a blue gradient
```
## 2. Visual exploration of categorical features
The second function covers the visual exploration of categorical features/columns. For this kind of EDA, the most common type of plot is the histogram, so I stuck with that for now. You can freely choose the number of columns (grid size) and the batch size of the output plots. Moreover, you can either choose a categorical feature as `key` or just type in `None` for a more general overview of the dataset, instead of a classification.
```python
def EDA_categorical(dataset, ncols:int, batch_size:int, dpi:int, key:str):
    """
    Visualizes the distribution of categorical columns in a loaded dataset, \n
    dataset = any dataframe, \n
    ncols = any number of columns you want the gridsize to be, \n
    batch_size = any number of figures you want to output at once, \n
    dpi = dots per inch resolution for plotting, \n
    key = any categorical feature you want to classify your data by.
    """
    try:
        import seaborn as sns
        print("Imported seaborn.")
        import matplotlib.pyplot as plt
        print("Imported matplotlib.")
        import math
        print("Imported math.")
    except:
        %pip install seaborn 
        import seaborn as sns
        print("Installed and imported seaborn.")
        %pip install matplotlib
        import matplotlib.pyplot as plt
        print("Installed and imported matplotlib.")
        %pip install math
        import math
        print("Installed and imported math.")

    # create lists with total number of numeric columns (int) and column names 
    n_features_col=dataset.select_dtypes(include=['object']).columns # name of columns with most common numeric dtypes
    n_features_int=len(n_features_col) # number of columns with most common numeric dtypes

    # create chunks for the plots
    for chunk_start in range(0, n_features_int, batch_size): # starts at 0, goes up to number of numerical features and increments by set batch_size
        chunk = n_features_col[chunk_start:chunk_start + batch_size] # slices the list into chunks defined by batch_size

        # specify figsize and dpi
        plt.figure(figsize=(10, 5), dpi=dpi)

        # loop through chunks
        for i, feature in enumerate(chunk, 1):  # Start from 1 for subplot index
            
            # calculates how many full rows are needed if you have ncols columns
            if n_features_int % 2 == 0: # if even:
                nrow_ncol=int(math.ceil(n_features_int) / ncols) # grid will fit nicely
            else: #if odd:
                nrow_ncol=int(math.ceil(n_features_int) / ncols)+1 # + 1 if odd

            # define subplot arrangement
            plt.subplot(nrow_ncol, ncols, i)

            # histogram
            if key is not None: # if classification (e.g., by 'species', 'id', ...) is given
                sns.histplot(data=dataset, x=feature, kde=False, hue=key, bins=50, stat='count', multiple='stack')
            else:
                sns.histplot(data=dataset, x=feature, kde=False, bins=50, stat='count', multiple='stack')
            plt.title(f'Distribution of {feature.capitalize()}')
            plt.xlabel(feature.capitalize())
            plt.ylabel('Count')

        plt.tight_layout()
        plt.show()
```
## 3. Visual exploration of numerical features
The third function allows you to explore the numerical features/column in the loaded dataset. It is similarly structured as above, but additionally includes the possibility to output different types of plots. I've added the most popular ones: histogram, boxplot, barplot and violinplot which can be controlled via `type`.
```python
def EDA_numerical(dataset, ncols:int, batch_size:int, dpi:int, key:str, type:str):
    """
    Visualizes the distribution of numerical columns in a loaded dataset. \n
    dataset = any dataframe, \n
    ncols = any number of columns you want the gridsize to be, \n
    batch_size = any number of figures you want to output at once, \n
    dpi = dots per inch resolution for plotting, \n
    key = any categorical feature you want to classify your data by, \n
    type = type of plot (i.e., 'hist', 'box', 'bar', 'violin') you want to output.
    """
    try:
        import seaborn as sns
        print("Imported seaborn.")
        import matplotlib.pyplot as plt
        print("Imported matplotlib.")
        import math
        print("Imported math.\n")
    except:
        %pip install seaborn 
        import seaborn as sns
        print("Installed and imported seaborn.")
        %pip install matplotlib
        import matplotlib.pyplot as plt
        print("Installed and imported matplotlib.")
        %pip install math
        import math
        print("Installed and imported math.\n")

    # create list with names of numeric columns and the total number of numeric features 
    n_features_col=dataset.select_dtypes(include=['int', 'float']).columns # name of columns with most common numeric dtypes
    n_features_int=len(n_features_col) # number of columns with most common numeric dtypes

    # create chunks/batches for the plots
    for chunk_start in range(0, n_features_int, batch_size): # starts at 0, goes up to number of numerical features and increments by set batch_size
        chunk = n_features_col[chunk_start:chunk_start + batch_size] # slices the list into chunks defined by batch_size

        # specify figsize and dpi
        plt.figure(figsize=(10, 16), dpi=dpi)

        # loop through chunks
        for i, feature in enumerate(chunk, 1):  # Start from 1 for subplot index

            # calculates how many full rows are needed if you have ncols columns
            if n_features_int % 2 == 0: # if even:
                nrow_ncol=int(math.ceil(n_features_int) / ncols) # grid will fit nicely
            else: #if odd:
                nrow_ncol=int(math.ceil(n_features_int) / ncols)+1 # + 1 if odd
            
            # define subplot arrangement
            plt.subplot(nrow_ncol, ncols, i)

            # entered type will device which plot to output
            # histogram
            if type == 'hist':
                if key is not None: 
                    sns.histplot(data=dataset, x=feature, kde=True, hue=key, bins=30)
                else: 
                    sns.histplot(dataset[feature], kde=True, bins=30)
                plt.title(f'Distribution of {feature.capitalize()}')
                plt.xlabel(feature.capitalize())
                plt.ylabel('Count')

            # boxplot
            elif type == 'box':
                if key is not None:
                    sns.boxplot(data=dataset, x=feature, y=key, hue=key)
                    plt.ylabel(key.capitalize())
                else: 
                    sns.boxplot(data=dataset, x=feature)
                    plt.yticks([])
                plt.title(f'Distribution of {feature.capitalize()}')
                plt.xlabel(feature.capitalize())

            # barplot
            elif type == 'bar':
                if key is not None: 
                    sns.barplot(data=dataset, x=feature, y=key, hue=key)
                    plt.ylabel(key.capitalize())
                else: 
                    sns.barplot(data=dataset, x=feature)
                    plt.yticks([])
                plt.title(f'Distribution of {feature.capitalize()}')
                plt.xlabel(feature.capitalize())

            # violin plot
            elif type == 'violin':
                if key is not None: 
                    sns.violinplot(data=dataset, x=feature, y=key, hue=key)
                    plt.ylabel(key.capitalize())
                else: 
                    sns.violinplot(data=dataset, x=feature)
                    plt.yticks([])
                plt.title(f'Distribution of {feature.capitalize()}')
                plt.xlabel(feature.capitalize())

            # type is not supported
            else:
                print(f"Type is not supported.")
        plt.tight_layout()
        plt.show()
```
## 4. Summary of statistical tests to find a fitting correlation coefficient 
Especially with high-dimensional datasets, we'd like to know how features correlate with each other. However, for a proper statistical evaluation, conditions must be met. 
The fourth function focusus on the output of useful information about the dataset (e.g., length, duplicate count, shapiro-wilk test statistics, AUC, ...) to check for normality and offer insights to decide which correlation coefficient (i.e., pearson, spearman, kendall) to apply. For that purpose, visual evaluations can help to support the decision whether or not the data follows a normal distribution. 
Next to density histograms and pp-plots, the function also includes a suggestion method for a good fitting coefficient, based on hard-coded conditions, which can be, of course, changed. The suggested classification gets pickled and can later on be used to process these informations.
```python
def correlation_summary(dataset, ncols:int, batch_size_per_feature:int, dpi:int, key:str):
    """
    Performs (statistical) tests to determine the right correlation coefficient and visualizes the relationship (i.e., correlation) of numercial features in a loaded dataset.\n
    dataset = any dataframe, \n
    ncols = any number of columns you want the gridsize to be, \n
    batch_size_per_feature = any number of figures you want to output at once (per feature), \n
    dpi = dots per inch resolution for plotting, \n
    key = any categorical feature you want to classify your data by.
    """
    try:
        import pandas as pd
        print("Imported pandas")
        import matplotlib.pyplot as plt
        print("Imported matplotlib.")
        import seaborn as sns
        print("Imported seaborn.")
        import scipy
        from scipy import stats
        print("Imported scipy and stats.")
        import math
        print("Imported math.")
        import numpy as np
        from numpy import trapezoid
        print("Imported numpy and trapezoid.")  
        import pylab
        print("Imported pylab.")
        import tabulate
        from tabulate import tabulate
        print("Imported tabulate")
    except:
        %pip install pandas 
        import pandas as pd
        print("Installed and imported pandas.")
        %pip install matplotlib
        import matplotlib.pyplot as plt
        print("Installed and imported matplotlib.")
        %pip install seaborn
        import seaborn as sns
        print("Installed and imported seaborn.")
        %pip install scipy
        import scipy
        from scipy import stats
        print("Installed, imported scipy and stats.")
        %pip install math
        import math
        print("Installed and imported math.")
        %pip install numpy
        import numpy as np
        from numpy import trapezoid
        ("Installed, imported numpy and trapezoid.")
        %pip install pylab
        import pylab
        print("Installed and imported pylab.")
        %pip install tabulate
        import tabulate
        from tabulate import tabulate
        print("Installed and imported tabulate")
    
    # summary of functions used to clean up the appearance and simplify 'correlation summary':

    # define function for a normal distribution layover
    def add_fit_to_histplot(a, fit=stats.norm, ax=None): # credits:https://stackoverflow.com/questions/64621456/plotting-a-gaussian-fit-to-a-histogram-in-displot-or-histplot

        if ax is None:
            ax = plt.gca()

        # compute bandwidth
        bw = len(a)**(-1/5) * a.std(ddof=1)
        # initialize PDF support
        x = np.linspace(a.min()-bw*3, a.max()+bw*3, 200)
        # compute PDF parameters
        params = fit.fit(a)
        # compute PDF values
        y = fit.pdf(x, *params)
        # plot the fitted continuous distribution
        ax.plot(x, y, color="#c44e52", linestyle='dashed')
        return ax
    
    # prepare dataset and perform statistics on it
    def perform_statistics(data):
        
        # only fetch numerical features from dataset
        dataset_num = data.select_dtypes(include=['int', 'float'])

        # create empty list for the results to be saved in
        results=[]

        # 1. loop through subsetted dataset with numerical data
        for feature in dataset_num:

            # test for normality
            stat_normal, p_value_normal = stats.normaltest(dataset_num[feature])
            stat_shapiro, p_value_shapiro = stats.shapiro(dataset_num[feature])
            
            # add pp-plot and calc the auc, compare with reference of linear reference line
            df_pp= dataset_num[feature]
            (prob_x, sample_y), (slope, intercept, r) = stats.probplot(df_pp, dist="norm")
            auc_dataset = trapezoid(sample_y, prob_x)

            # Predicted y values from the pp line
            ref_y = slope * prob_x + intercept
            (prob_x_ref, sample_y_ref), (slope_ref, intercept_ref, r_ref) = stats.probplot(ref_y, dist="norm")
            auc_ref = trapezoid(sample_y_ref, prob_x_ref)
            
            # difference between auc dataset and auc reference (%)
            auc_diff = round(((auc_dataset-auc_ref)/auc_ref)*100, 2)
            if auc_diff < 0:
                auc_diff = auc_diff*-1

            # Mean Squared Deviation (MSD)
            msd_absolute = np.mean((sample_y - ref_y) ** 2) # mean(sqrt(observed data-expected data))

            # check for length of features
            feature_n = dataset_num[feature].notna().size

            # check for duplicate values, fetch counts and calc the difference to feature_n (%)
            duplicates = dataset_num[feature].duplicated().value_counts() # outputs True and False with respective count values
            raw_duplicate_count = duplicates.get(True, 0) # only gets true values, if not true -> returns 0 
            raw_duplicate_percent = round((raw_duplicate_count/feature_n)*100, 2) # percentage of duplicates for each feature

            results.append({    
                "feature": feature,
                "feature: n": feature_n,
                "duplicates absolute": raw_duplicate_count,
                "duplicates %": raw_duplicate_percent,
                "statistic normal": stat_normal,
                "statistic shapiro": stat_shapiro,
                "p-value normal": p_value_normal,
                "p-value shapiro": p_value_shapiro,
                "auc data": auc_dataset,
                "auc reference": auc_ref,
                "auc diff": auc_diff,
                "msd": msd_absolute
            })

        # print out df with saved information
        results_df = pd.DataFrame(results)
        print(f"\nSummary of statistical tests to determine best fitting correlation metric.")
        headers=['Feature: name', 'Feature: n', 'Duplicates: count', 'Duplicates: %', 'Normaltest: statistic', 'Shapiro Wilk: statistic', 'Normaltest: p-value', 'Shapiro Wilk: p-value', 'AUC: loaded dataset', 'AUC: reference line', 'AUC: absolute difference (%)', 'Mean Squared Deviation (MSD)']
        print(tabulate(results_df.values.tolist(),headers=headers, tablefmt='rounded_grid'))
        
        # 2. assign correlation coefficient to features
        for index, row in results_df.iterrows(): # iterate through rows of results_df to find features which meet defined conditions

            score=0

            # define conditions for suggested categorization of correlation coefficients
            condition_auc = row['auc diff'] <= 3.5  # if smaller or equal to 3.5, high possibility of being normal distributed
            condition_msd = row['msd'] <= 0.5 # if smaller or equal to 0.5, high possibility of being normal distributed
            condition_n = row['feature: n'] > 30 # if length of feature is greater than 30
            condition_ties = row['duplicates %'] < 20 # if less than 20 % of duplicates/tied ranks
            condition_skew_kurtosis = row['statistic normal'] < 55 # if kurtosis+skew score is smaller than 55

            # add a weight to condtions
            if condition_msd:
                score += 2
            if condition_auc:
                score += 2
            if condition_skew_kurtosis:
                score += 1
            if condition_n:
                score += 0.25
            if condition_ties:
                score += 0.25
            
            # append to lists
            if key is not None:
                if score >= 5:
                    pearson.append({"feature": row['feature'], "subset": value})
                elif score >= 0.5:
                    spearman.append({"feature": row['feature'], "subset": value})
                else:
                    kendall.append({"feature": row['feature'], "subset": value})
            else:
                if score >= 5:
                    pearson.append({"feature": row['feature']})
                elif score >= 0.5:
                    spearman.append({"feature": row['feature']})
                else:
                    kendall.append({"feature": row['feature']})

    # prepare dataset and plot histogram + pp-plot
    def plot_hist_pp(data):

        # prepare subset for plotting
        n_features_col = data.select_dtypes(include=['int', 'float']).columns # selects only numeric features
        n_features_int = len(n_features_col) # length of numeric features

        # separat subsetted data into chunks of batch_size_per_feature  
        for chunk_start in range(0, n_features_int, batch_size_per_feature): # starts at 0, goes up to number of numerical features and increments by set batch_size
            chunk = n_features_col[chunk_start:chunk_start + batch_size_per_feature] # slices the list into chunks defined by batch_size
            n_features = len(chunk)
            
            # set some parameters
            n_subplots = n_features * 2  # hist + pp-plot for each feature
            nrows = math.ceil(n_subplots / ncols)
            plt.figure(figsize=(10, 6), dpi=dpi)

            # plot histogram and pp-plot per feature
            for i, feature in enumerate(chunk):

                # histogram
                plt.subplot(nrows, ncols, 2 * i + 1)
                sns.histplot(data[feature], kde=True, bins=50, stat='density')
                add_fit_to_histplot(data[feature], fit=stats.norm)
                plt.title(f'Distribution of {feature.capitalize()}')
                plt.xlabel(feature.capitalize())
                plt.ylabel('Density')

                # pp-plot
                plt.subplot(nrows, ncols, 2 * i + 2)
                stats.probplot(data[feature], dist="norm", plot=pylab)
                plt.title(f'P-P plot of {feature.capitalize()}')

            plt.tight_layout()
            plt.show()


    # start with pipeline

    # if key is provided:
    if key is not None:
        
        # 1. create empty lists for the results to be saved in
        pearson = []
        spearman = []
        kendall = []

        # 2. get unique names of provided key
        class_names = dataset[key].unique()

        # 3. subset data based on provided key
        for value in class_names:
            subset = dataset[dataset[key] == value]
            print(f"\nPerformed subsetting for '{value}' of provided key '{key}'.")
        
            # 4. prepare datset and perform statistics on it + assign features to correlation coefficient based on met conditions
            perform_statistics(data=subset)

            # 5. prepare dataset and plot histogram+pp-plot with reference curve+line to display normal distribution
            plot_hist_pp(data=subset)
            
        # 6. convert filled lists to df, ignore empty lists and print out dfs
        if len(pearson) == 0 and len(spearman) == 0 and len(kendall) == 0:
            print("Lists are empty. No classification could be performed.")
        if len(pearson) != 0:
            pearson_df = pd.DataFrame(pearson)
            pearson = pd.to_pickle(pearson_df, 'pearson.pkl')
            print("\nSuggested correlation coefficient: pearson's r.")
            print(tabulate(pearson_df.values.tolist(),headers=['Feature: name', 'Subset: name'], tablefmt='rounded_grid'))
        if len(spearman) !=0:
            spearman_df = pd.DataFrame(spearman)
            spearman = pd.to_pickle(spearman_df, 'spearman.pkl')
            print("\nSuggested correlation coefficient: spearman's ρ.")
            print(tabulate(spearman_df.values.tolist(),headers=['Feature: name', 'Subset: name'], tablefmt='rounded_grid'))
        if len(kendall) != 0:
            kendall_df = pd.DataFrame(kendall)
            kendall = pd.to_pickle(kendall_df, 'kendall.pkl')
            print("\nSuggested correlation coefficient: kendall's τ.")
            print(tabulate(kendall_df.values.tolist(),headers=['Feature: name', 'Subset: name'], tablefmt='rounded_grid'))


    # if no key is provided  
    else:

        # 1. create empty lists for the results to be saved in
        pearson = []
        spearman = []
        kendall = []

        # 2. prepare datset and perform statistics on it + assign features to correlation coefficient based on met conditions
        perform_statistics(data=dataset)

        # 3. prepare dataset and plot histogram+pp-plot with reference curve+line to display normal distribution
        plot_hist_pp(data=dataset)

        # 4. convert filled lists to df, ignore empty lists and print out dfs
        if len(pearson) == 0 and len(spearman) == 0 and len(kendall) == 0:
            print("Lists are empty. No classification could be performed.")
        if len(pearson) != 0:
            pearson_df = pd.DataFrame(pearson)
            pearson = pd.to_pickle(pearson_df, 'pearson.pkl')
            print("\nSuggested correlation coefficient: pearson's r.")
            print(tabulate(pearson_df.values.tolist(),headers=['Feature: name'], tablefmt='rounded_grid'))
        if len(spearman) !=0:
            spearman_df = pd.DataFrame(spearman)
            spearman = pd.to_pickle(spearman_df, 'spearman.pkl')
            print("\nSuggested correlation coefficient: spearman's ρ.")
            print(tabulate(spearman_df.values.tolist(),headers=['Feature: name'], tablefmt='rounded_grid'))
        if len(kendall) != 0:
            kendall_df = pd.DataFrame(kendall)
            kendall = pd.to_pickle(kendall_df, 'kendall.pkl')
            print("\nSuggested correlation coefficient: kendall's τ.")
            print(tabulate(kendall_df.values.tolist(),headers=['Feature: name'], tablefmt='rounded_grid'))
```
## 5. Processing statistical information and visualize relationships
The last function processes the information collected in `correlation_summary` and outputs a correlation matrix, either using the correlation coefficient of your choice or, by entering `corr='auto'`, the suggested classification. When using the `auto` function and a `key`, make sure to use the same `key` as used before in `correlation_summary` for the correct data being used.
```python
def correlation_visualization(dataset, corr:str, dpi:int, key:str): 
    """
    Visualizes the relationship (i.e., correlation) of numercial features in a loaded dataset,\n
    dataset = any dataframe, \n
    corr = correlation coefficient (i.e., 'pearson', 'spearman', 'kendall'); 'auto' = takes suggested classification by the function 'correlation_summary', \n
    dpi = dots per inch resolution for plotting, \n
    key = any categorical feature you want to classify your data by.
    """
    try:   
        import pandas as pd
        print("Imported pandas")
        import seaborn as sns
        print("Imported seaborn.")
        import matplotlib.pyplot as plt
        print("Imported matplotlib.")
        from tabulate import tabulate
    except:
        %pip install seaborn
        import seaborn as sns
        print("Installed and imported seaborn.")
        %pip install matplotlib
        import matplotlib.pyplot as plt
        print("Installed and imported matplotlib.")

    # set some visualization parameters
    sns.set_theme(style="whitegrid") # appearance
    plt.figure(figsize=(10, 5), dpi=dpi) 

    # overview of functions used:

    # set up 'corr' + respective suffix
    def determine_suffix(coefficient:str):
        if coefficient == 'pearson':
            suffix = "'s r"
        elif coefficient == 'spearman':
            suffix = "'s ρ"
        else:
            suffix = "'s τ"
        return coefficient+suffix
    
    # prepare dataset and plot correlation matrix:
    def prepare_and_plot(data, coefficient:str):

        # call function 'determine_suffix' to create the full name of the used correlation coefficient
        corr_full_name = determine_suffix(df_name if corr == 'auto' else corr)

        # select only numerical columns
        numeric_df = data.select_dtypes(include=['int', 'float'])

        # Compute the correlation matrix using the specified method
        corr_mat = numeric_df.corr(method=coefficient).stack().reset_index(name=coefficient)

        # Create the plot
        rel = sns.relplot(
            data=corr_mat,
            x="level_0", y="level_1", hue=coefficient, size=coefficient,
            palette="vlag", hue_norm=(-1, 1), edgecolor=".5",
            height=10, sizes=(50, 300), size_norm=(-.2, .8)
        )
        rel.set(xlabel="", ylabel="", aspect="equal")
        rel.despine(left=True, bottom=True)

        # add title if key is not None for more information about the used dataset 
        if key is not None:
            plt.title(f"{corr_full_name.capitalize()} applied on categorical feature '{key.capitalize()}' with subset of '{class_name.capitalize()}'")

        rel._legend.set_title(f"{corr_full_name.capitalize()}") # adjust legend title 
        plt.subplots_adjust(right=0.89) # leave some space for the legend title
        rel.ax.margins(.02)
        for label in rel.ax.get_xticklabels():
            label.set_rotation(90)
            label.set_horizontalalignment('right')
        plt.show()
        
    # if no key is provided:
    if key is None:

        # if no correlation coefficient is entered, but auto categorization is desired
        if corr == 'auto':

            # correlation coefficients
            dfs_list = ['pearson', 'spearman', 'kendall']

            # create empty dict for the data to be saved into
            dfs_dict = {}

            # 1. loop through list of correlation coefficients and load specific .pkl data 
            for df_name in dfs_list:

                try:
                    # loads pickled dataframes with suggested correlation coefficient for features in dataset
                    df_features = dfs_dict[df_name] = pd.read_pickle(f'{df_name}.pkl')
                    print(f"\nLoaded {df_name}.pkl:")
                    print(tabulate(df_features, tablefmt='rounded_grid'))

                    # gets values (feature names) from dict
                    values = df_features['feature']

                except Exception as e:
                    print(f"\nCould not load {df_name}.pkl: {e}")
                
                # create subsets
                try:
                    subset = dataset[values]
                except KeyError as e:
                    print(f"{df_name} not found in dataset: {e}")

                # prepare dataset and plot correlation matrix
                prepare_and_plot(data=subset, coefficient=df_name)

        # if correlation coefficient ('pearson', 'spearman', 'kendall') is provided
        else:

            # prepare datasets and plot correlation matrix 
            prepare_and_plot(data=dataset, coefficient=corr)


    # if key is provided:
    else:

        # if auto categorization of correlation coefficients:
        if corr == 'auto':

            # correlation coefficients
            dfs_list = ['pearson', 'spearman', 'kendall']

            # create empty dict for the data to be saved into
            dfs_dict = {}

            # 1. loop through list and load specific .pkl data 
            for df_name in dfs_list:

                try:
                    # loads pickled dataframes with suggested correlation coefficient for features in dataset
                    df_features = dfs_dict[df_name] = pd.read_pickle(f'{df_name}.pkl')
                    print(f"\nLoaded {df_name}.pkl:")
                    print(tabulate(df_features, tablefmt='rounded_grid'))

                    # gets values (feature names) from dict
                    values = df_features['feature']

                except Exception as e:
                    dfs_dict[df_name] = None
                    print(f"\nCould not load {df_name}.pkl: {e}")
                
                # Subset the dataset by each class in the key
                try:
                    class_names = dataset[key].unique().tolist()
                except KeyError as e:
                    print(f"'{key}' not found in dataset: {e}")
                
                # 2. loop through subset
                for class_name in class_names:
                    subset_df = dataset[dataset[key] == class_name]
                    selected_columns = subset_df[values]

                    try:
                        prepare_and_plot(data=selected_columns, coefficient=df_name)
                    except Exception as e:
                        print(f"Error processing '{class_name}' for correlation coefficient '{corr}': {e}")

        # if correlation coefficent is provided
        else:

            # get unique key names
            class_names = dataset[key].unique().tolist()
            class_names

            # create empty dict for the dfs to be saved into
            keys_df={}

            # create subset for each unique class name
            for class_name in class_names:
                df = dataset[dataset[key] == class_name]
                keys_df[class_name] = df
            
            # for the class name and associated data, get items and create correlation plot
            for class_name, df in keys_df.items():

                # prepare datasets and plot correlation matrix 
                prepare_and_plot(data=dataset, coefficient=corr) 
```
