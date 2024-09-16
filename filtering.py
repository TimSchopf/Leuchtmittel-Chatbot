def filter_dataframe(df, filters):
    """
    Filters a DataFrame based on specified conditions and returns the result as a list of dictionaries.

    Parameters:
    df (pd.DataFrame): The DataFrame to filter.
    filters (dict): A dictionary where keys are column names and values are the conditions.
                    Conditions can be:
                    - A dictionary with 'operator' and 'value' keys
                    - A direct value for equality check
                    - None to skip filtering for that column

    Returns:
    list or None: The filtered DataFrame as a list of dictionaries or None if the result is empty.
    """
    filtered_df = df

    for column, condition in filters.items():
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        if condition is None:
            continue

        if isinstance(condition, dict) and 'operator' in condition and 'value' in condition:
            operator = condition['operator']
            value = condition['value']

            if operator not in ['>', '<', '>=', '<=', '==', '!=']:
                raise ValueError(f"Operator '{operator}' is not supported.")

            if operator == '>':
                filtered_df = filtered_df[filtered_df[column] > value]
            elif operator == '<':
                filtered_df = filtered_df[filtered_df[column] < value]
            elif operator == '>=':
                filtered_df = filtered_df[filtered_df[column] >= value]
            elif operator == '<=':
                filtered_df = filtered_df[filtered_df[column] <= value]
            elif operator == '==':
                filtered_df = filtered_df[filtered_df[column] == value]
            elif operator == '!=':
                filtered_df = filtered_df[filtered_df[column] != value]
        else:
            # Direct value comparison
            if isinstance(df[column].iloc[0], list):
                # For list columns, check if the value is in the list
                filtered_df = filtered_df[filtered_df[column].apply(lambda x: condition in x)]
            else:
                filtered_df = filtered_df[filtered_df[column] == condition]

    return filtered_df.to_dict('records') if not filtered_df.empty else None