# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import inline as inline
# import matplotlib
import numpy as np
import pandas as pd
import seaborn
import seaborn as sb
from pandas import Series, DataFrame
from numpy.random import randn
import matplotlib.pyplot as plt
from pylab import rcParams
from pandas.plotting import scatter_matrix


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
    print('=================================================')
    print(DF_obj2)
    print(DF_obj.loc[['row 2', 'row 5'], ['column 5', 'column 2']])

    print(series_obj['row3':'row7'])

    print('*************************************************')
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
    print('------------------------------')
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
    address = 'D:/Linkedin Learning/Python for Data Science Essential Training Part ' \
              '1/Ex_Files_Python_Data_Science_EssT_Pt_1/Exercise Files/Data/mtcars.csv'
    cars = pd.read_csv(address)
    cars.columns = ['Car_Names', 'MPG', 'Cyl', 'Disp', 'HP', 'Drat', 'WT', 'QSec', 'VS', 'AM', 'Gear', 'Crab']
    print(cars.head())
    carsGroupsCyl = cars.groupby(cars['Cyl'])
    carsGroupsAM = cars.groupby(cars['AM'])
    print(carsGroupsCyl.mean())
    print(carsGroupsAM.mean())
    print(cars.all())
    x = range(1, 10)
    # y = [1, 2, 3, 4, 0, 4, 3, 2, 1]
    y = [1, 2, 3, 4, 0.5, 4, 3, 2, 1]
    x1 = range(0, 10)
    y1 = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    z = [1, 2, 3, 4, 0.5]
    wide = [.5, .5, .5, .9, .9, .9, .5, .5, .5]
    # wide = [.1, .3, .5, .7, .9, .7, .5, .3, .1]
    color = ['salmon']
    colorThem = ['darkgray', 'lightsalmon', 'powderblue']
    colorThemRGB = ['#A9A9A9', '#FFA07A', '#B0E0E6', '#FFE4C4', '#BDB76B']

    # plt.plot(x, y)
    # plt.show()
    # plt.bar(x, y)
    # plt.show()
    # plt.pie(z)
    # plt.savefig('E:/newTest/pie_chart.png')
    # plt.show()
    # mpg = cars['MPG']
    # mpg.plot()
    # plt.show()
    dataFramePlat1 = cars[['Cyl', 'WT', 'MPG']]
    # dataFramePlat1.plot()
    # plt.show()
    # mpg.plot(kind='bar')
    # plt.show()
    # mpg.plot(kind='barh')
    # plt.show()

    rcParams['figure.figsize'] = 5, 4
    fig = plt.figure()
    ax = fig.add_axes([.1, .1, 1, 1])
    # ax.set_xlim([1, 9])
    # ax.set_ylim([0, 5])
    # ax.set_xticks([0, 1, 2, 4, 5, 6, 8, 9, 10])
    # ax.set_yticks([0, 1, 2, 3, 4, 5])
    # ax.grid()
    # ax.plot(x, y)
    # plt.show()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(x)
    # ax2.plot(x, y)
    # plt.show()
    # plt.bar(x, y, color=color, width=wide, align='center')
    # plt.show()
    # dataFramePlat1.plot(color=colorThem)
    # plt.show()
    # plt.pie(z, colors=colorThemRGB)
    # plt.show()
    # plt.plot(x, y, ds='steps', lw=5)
    # plt.plot(x1, y1, ls='--', lw=10)
    # plt.plot(x, y, marker='1', mew=20)
    # plt.plot(x1, y1, marker='+', mew=15)
    # plt.show()
    # plt.bar(x, y)
    # plt.xlabel('Your x-axis label')
    # plt.ylabel('Your y-axis label')
    # plt.show()
    # vehicleType = ['Bicycle', 'Motorbike', 'Car', 'Van', 'Stroller']
    # plt.pie(z, labels=vehicleType)
    # plt.legend(vehicleType, loc='best')
    # plt.show()
    # mpgCars = cars.MPG
    # mpgCars.plot()
    # ax.set_xticks(range(32))
    # ax.set_xticklabels(cars.Car_Names, rotation=60, fontsize='medium')
    # ax.set_title('Miles per Gallon of Cars in mtCars Dataset')
    # ax.set_xlabel('Car Names')
    # ax.set_ylabel('Miles / Gal')
    # ax.legend(loc='best')
    # ax.set_ylim([0, 45])
    # ax.annotate('Toyota Corolla', xy=(19, 33.9), xytext=(21, 35), arrowprops=dict(facecolor='black', shrink=0.05))
    # plt.show()
    # print(mpgCars.max())
    # plt.savefig('E:/newTest/pie_chart.png')
    addressTime = 'D:/Linkedin Learning/Python for Data Science Essential Training Part ' \
                  '1/Ex_Files_Python_Data_Science_EssT_Pt_1/Exercise Files/Data/Superstore-Sales.csv'
    dataFrameTime = pd.read_csv(addressTime, index_col='Order Date', encoding='cp1252', parse_dates=True)
    pd.set_option('display.max_columns', None)
    print(dataFrameTime.head())
    # dataFrameTime['Order Quantity'].plot()
    # plt.show()
    dataFrameTimeSample = dataFrameTime.sample(n=100, random_state=25, axis=0)
    dataFrameSubsetCars = cars[['MPG', 'Disp', 'HP', 'WT']]
    # plt.xlabel('Order Date')
    # plt.ylabel('Order Quantity')
    # plt.title('Superstore Sales')
    # dataFrameTimeSample['Order Quantity'].plot()
    # plt.show()
    # print(dataFrameTime.columns.tolist())
    sb.set_style('whitegrid')
    cars.index = cars.Car_Names
    mpg = cars['MPG']
    # mpg.plot(kind='hist')
    # plt.show()
    # plt.hist(mpg)
    # plt.plot()
    # plt.show()
    # sb.displot(mpg)
    # plt.show()
    # cars.plot(kind='scatter', x='HP', y='MPG', c=['darkgray'], s=150)
    # plt.show()
    # sb.regplot(x='HP', y='MPG', data=cars, scatter=True)
    # sb.pairplot(cars)
    # sb.pairplot(dataFrameSubsetCars)
    # plt.show()
    # cars.boxplot(column='MPG', by='AM')
    # cars.boxplot(column='WT', by='AM')
    # plt.show()
    sb.boxplot(x='AM', y='MPG', data=cars, palette='hls')
    plt.show()

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
