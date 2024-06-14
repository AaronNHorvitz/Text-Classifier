import pandas as pd

def display_categories_and_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Displays a summary of categories and their associated key terms from a DataFrame containing
    categorized document information. This function assumes each category in the DataFrame has consistent
    key terms across different entries.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame that must include at least two columns: 'category' and 'key_terms'.
        - 'category': A column containing category identifiers.
        - 'key_terms': A column containing a string or list of key terms associated with each category.

    Returns:
    -------
    pd.DataFrame
        A new DataFrame where each row corresponds to a unique category and its associated key terms.

    Examples:
    --------
    >>> df = pd.DataFrame({
        'category': ['Technology', 'Technology', 'Health', 'Health'],
        'key_terms': ['AI, Machine Learning', 'AI, Machine Learning', 'Disease Prevention', 'Disease Prevention']
    })
    >>> categories_df = display_categories_and_terms(df)
    >>> print(categories_df)

    Notes:
    -----
    This function is particularly useful for quickly summarizing the categorical breakdown of document datasets,
    especially when each category has been assigned specific key terms. The function simplifies the visualization
    of these relationships in a tabular form.
    """
    # Get unique categories
    unique_categories = df["category"].unique()

    # Create a display DataFrame
    categories_df = pd.DataFrame(columns=["Category", "Key Terms"])

    for category in unique_categories:
        # Get the key terms for the current category
        key_terms = df[df["category"] == category]["key_terms"].iloc[
            0
        ]  # Assuming key terms are consistent within each category

        # Append to the display DataFrame
        categories_df = categories_df.append(
            {"Category": category, "Key Terms": key_terms}, ignore_index=True
        )

    # Display the DataFrame
    return categories_df
