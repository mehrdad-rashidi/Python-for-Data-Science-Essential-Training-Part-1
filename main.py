# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
from pandas import Series, DataFrame


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print_hi('PyCharm')
    series_obj = Series(np.arange(8), index=['row1', 'row2', 'row3', 'row4', 'row5', 'row6', 'row7', 'row8'])
    series_obj1 = Series(['row_1', 'row_2', None, 'row_4', 'row_5', 'row_6', None, 'row_8'])
    print(series_obj1)
    print(series_obj)
    print(series_obj['row7'])
    print(series_obj[[0, 7]])
    np.random.seed(25)
    DF_obj = DataFrame(np.random.rand(36).reshape((6, 6)), index=['row 1', 'row 2', 'row 3', 'row 4', 'row 5', 'row 6'],
                       columns=['column 1', 'column 2', 'column 3', 'column 4', 'column 5', 'column 6'])
    DF_obj1 = DataFrame(np.random.rand(36).reshape((6, 6)))
    DF_obj2 = DataFrame(np.random.rand(36).reshape((6, 6)))
    DF_obj2.loc[3:5, 0] = None
    DF_obj2.loc[1:4, 5] = None
    print(DF_obj)
    print('*************************************************')
    print(DF_obj2)
    print(DF_obj.loc[['row 2', 'row 5'], ['column 5', 'column 2']])

    print(series_obj['row3':'row7'])

    print(DF_obj < .2)
    print(series_obj[series_obj > 6])
    series_obj['row1', 'row5', 'row8'] = 8
    print(series_obj)
    print(series_obj1)
    print(series_obj1.isnull())
    print(DF_obj1)
    DF_obj1.loc[3:5, 0] = None
    DF_obj1.loc[1:4, 5] = None
    print(DF_obj1)
    DF_obj1_missing_fill = DF_obj1.fillna(0)
    print(DF_obj1_missing_fill)
    DF_obj1_missing_fill = DF_obj1.fillna({0: 0.1, 5: 1.25})
    print(DF_obj1_missing_fill)
    DF_obj1_missing_fill = DF_obj1.fillna(method='ffill')  # fill the missing value with last not null value element
    print(DF_obj1_missing_fill)
    print(DF_obj2)
    print('------------------------------')
    print(DF_obj2.isnull().sum())
    DF_obj2_Not_Missing_Value_rows = DF_obj2.dropna()
    DF_obj2_Not_Missing_Value_column = DF_obj2.dropna(axis=1)
    print('------------------------------')
    print(DF_obj2_Not_Missing_Value_column)
    print('------------------------------')
    print(DF_obj2_Not_Missing_Value_rows)  # omitted the row of has missing value
    DF_obj3 = DataFrame({'column 1': [1, 1, 2, 2, 3, 3, 3],
                         'column 2': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                         'column 3': ['A', 'A', 'B', 'B', 'C', 'C', 'C']})
    DF_obj4 = DataFrame({'column 1': [1, 1, 2, 2, 3, 3, 3],
                         'column 2': ['a', 'a', 'b', 'b', 'c', 'c', 'c'],
                         'column 3': ['A', 'A', 'B', 'B', 'D', 'C', 'C']})
    print('------------------------------')
    print(DF_obj3)
    print('------------------------------')
    print(DF_obj3.duplicated())
    DF_obj3_drop_duplicate = DF_obj3.drop_duplicates()
    DF_obj4_drop_duplicate_col = DF_obj4.drop_duplicates(['column 3'])
    print('------------------------------')
    print(DF_obj3_drop_duplicate)
    print('------------------------------')
    print(DF_obj4_drop_duplicate_col)
    dataFrameObject5 = pd.DataFrame(np.arange(36).reshape(6, 6))
    dataFrameObject6 = pd.DataFrame(np.arange(15).reshape(5, 3))
    dataFrameObject7 = pd.concat([dataFrameObject5, dataFrameObject6], axis=1)  # concatenate by column
    dataFrameObject8 = pd.concat([dataFrameObject5, dataFrameObject6])  # concatenate by row
    print('------------------------------')
    print(dataFrameObject5)
    print('------------------------------')
    print(dataFrameObject6)
    print('------------------------------')
    print(dataFrameObject7)
    print('------------------------------')
    print(dataFrameObject8)
    dataFrameObject5.drop([0, 2])  # Drop rows 0 and 2
    dataFrameObject5.drop([0, 2], axis=1)  # Drop columns 0 and 2
    seriesObject = Series(np.arange(6))
    seriesObject.name = "addedVariable"
    addedVariable = DataFrame.join(dataFrameObject5, seriesObject)
    print('------------------------------')
    print(addedVariable)
    addedDataTable = addedVariable.append(addedVariable,
                                          ignore_index=False)  # The frame.append method is deprecated and will be
    # removed from pandas in a future version. Use pandas.concat instead.
    print('------------------------------')
    print(addedDataTable)
    addedDataTable1 = addedVariable.append(addedVariable,
                                           ignore_index=True)  # The frame.append method is deprecated and will be
    # removed from pandas in a future version. Use pandas.concat instead.
    print('------------------------------')
    print(addedDataTable1)
    dataFameSorted = dataFrameObject5.sort_values(by=5, ascending=False)
    print('------------------------------')
    print(dataFameSorted)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
