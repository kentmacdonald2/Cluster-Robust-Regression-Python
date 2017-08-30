import pandas as pd
import statsmodels.api as sm


def import_data(file):
    # import just the "math-class" data
    return pd.read_csv(file, delimiter=";")

if __name__ == '__main__':
    filepath = 'data/student-mat.csv'

    # load in the dataset
    data = import_data(filepath)

    # print the first few lines of the dataset to ensure it is imported correctly
    print(data.head())

    # exogenous variable (# of absences)
    x = data['absences']

    # endogenous variable (final grade)
    y = data['G3']

    # add a constant to get the intercept
    x = sm.add_constant(x)

    # create the linear regression model
    model = sm.OLS(exog=x, endog=y)

    # fit the model with clustering on school
    results = model.fit(cov_type='cluster', cov_kwds={'groups': data['school']}, use_t=True)

    # print regression results
    print(results.summary())
