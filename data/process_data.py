import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets from cvs files
    Merge datasets
    Args:
        messages_filepath (string): filepath of the messages dataset
        categories_filepath (string): filepath of the categories dataset
    Returns:
        df (pandas dataframe): merged dataframe (on column id)
            of messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge datasets on column id
    df = messages.merge(categories, how='inner', on='id')
    return df



def clean_data(df):
    """
    Clean merged dataframe of categories and mesages:
        Split the values in the categories column on the ; character so that
        Use the first row of categories dataframe to create column names for
        the categories data.each value becomes a separate column.
        Rename columns of categories with new column names.
        Drop duplicates
        Iterate through the category columns in df to keep only the last character
        of each string (the 1 or 0). For example, related-0 becomes 0, related-1
        becomes 1. Convert the string to a numeric value
        Drop the categories column from the df dataframe since it is no longer needed.
        Concatenate df and categories data frames.
        Drop duplicated rows
        Drop outliers (or anomalies) rows with value=2 in column 'related'

        Args: df (pandas dataframe)

        Returns: df (pandas dataframe)


    """
    # create a dataframe of the 36 individual category columns
    category_df = df.categories.str.split(';', expand=True)

    row = df.loc[0,'categories']
    category_colnames = pd.Series(row.split(';')).apply(lambda x : x[:-2]).values
    # rename the columns of `categories`
    category_df.columns = category_colnames
    # Drop this ones?
    # category_df[categories.related=='related-2']
    for column in category_df:
        # set each value to be the last character of the string
        category_df[column] = category_df[column].str[-1]
        # convert column from string to numeric
        category_df[column] = category_df[column].astype(int)
    #Drop "categories" column from original dataframe
    df.drop('categories', inplace=True, axis=1)
    # concatenate the original dataframe with the new dataframe
    df = pd.concat([df, category_df], axis=1)
    # Drop duplicates
    df.drop_duplicates(subset='id', keep='first', inplace=True)
    # remove some dirty data (188 rows on "related" label with value == 2 )
    df = df[df.related!=2]

    return df


def save_data(df, database_filename):
    """
    Saving clean dataset into an sqllite database. The data will be saved under
    the 'data' directory with column name 'Response'

    Args:
        df (pandas dataframe): dataframe to save
        database_filename (string): name of the database

    Returns:
    """

    engine = "sqlite:///"+database_filename
    create_engine(engine)
    df.to_sql('Response', engine, index=False, if_exists='replace')


def main():
    """
    Parse command-line arguments: expected 3 arguments:

    Example: python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

    Args:
        messages dataset file (string)
        categories dataset file (string)
        database name to save (string)
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
