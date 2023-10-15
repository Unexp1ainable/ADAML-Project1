import pandas as pd


def reorder_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df[["% Iron Feed", "% Silica Feed", "Starch Flow", "Amina Flow", "Ore Pulp Flow", 'Ore Pulp pH', 'Ore Pulp Density', 
                'Flotation Column 01 Air Flow', 'Flotation Column 01 Level',
                'Flotation Column 02 Air Flow', 'Flotation Column 02 Level',
                'Flotation Column 03 Air Flow', 'Flotation Column 03 Level',
                'Flotation Column 04 Air Flow', 'Flotation Column 04 Level',
                'Flotation Column 05 Air Flow', 'Flotation Column 05 Level',
                'Flotation Column 06 Air Flow', 'Flotation Column 06 Level',
                'Flotation Column 07 Air Flow', 'Flotation Column 07 Level'
                ]]


def shift(df_x: pd.DataFrame, df_y: pd.DataFrame, step: int, column_begin: str, column_end : str | None = None):
    index_begin = df_x.columns.get_loc(column_begin)
    index_end = None

    if column_end:
        index_end = df_x.columns.get_loc(column_end) + 1

    df_x.iloc[:, index_begin:index_end] = df_x.iloc[:, index_begin:index_end].shift(-step)
    if step > 0:
        df_x.drop(df_x.tail(step).index, inplace = True)
        df_y.drop(df_y.tail(step).index, inplace=True)
    else:
        df_x.drop(df_x.head(step).index, inplace = True)
        df_y.drop(df_y.head(step).index, inplace=True)

