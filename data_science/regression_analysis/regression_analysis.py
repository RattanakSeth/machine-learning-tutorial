import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

class RegressionAnalysis:
    """
        This question involves the use of simple linear regression on the Auto data set.
            (a) Use the sm.OLS() function to perform a simple regression with mpg as the response and
            horsepower as the predictor. Use the summarize() function to print the results. Comment
            on the output. For example:
                (i) Is there a relationship between the predictor and the response?
                (ii) How strong is the relationship between the predictor and the response?
                (iii) Is the relationship between the predictor and the response positive or negative?
                (iv) What is the predicted mpg associated with a horsepower of 98? What are the
                associated 95% and prediction intervals?
            (b) Plot the response and the predictor in a new set of axes ax. Use the ax.axline() method
            or the abline() function defined in the lab to display the least squares
            (c) Produce some of the diagnostic plots of the least square regression fit as described in the
            lab. Comment on any problems you see with the fit. regression line.
            Master of Computer Science Page 2
    """
    def simpleLinearRegressionOnAutoDataset():
        # Step 1: Load data
        # Load the Auto dataset (assuming it is a CSV file)
        # Ref dataset: https://github.com/selva86/datasets/blob/master/Auto.csv
        autoData = pd.read_csv('data/auto/Auto.csv')
        # print(autoData)

        # Prepare the data
        X = autoData['horsepower']
        Y = autoData['mpg']
        print("\n============= X: horsepower =================\n")
        print(X)
        print("\n============= End of X =================\n")

        # Add a constant to the predictor variable (for the intercept term)
        X = sm.add_constant(X)
        print("\n==========Add Constant to X=================\n")
        print(X)
        print("\n==============End of applying================\n")

        # Step 2: Perform the linear regression
        # Perform the linear regression
        model = sm.OLS(Y, X).fit()

        # Print the summary
        print("\n=================================Summary of the model===============================\n")
        print(model.summary())
        print("\n=================================End of Summary of the model==========================\n")

        # Step 3: Interpret the output
        """
            1. Relationship between predictor and response
                . Check the p-value of the`horsepower` coefficient. If it is less than 0.05, there is a significant relationship.
            2. Strength of the relationship:
                . Look at the R-squared value. It indicates the proportion of variance in `mpg` (miles per gallon) explained by `horsepower`
            3. Positive or negative relationship:
                . Examine the sign of the `horsepower` coefficient. A negative coefficient indicates a negative relationship, and vice versa.
        """
        # Prediction and intervals for `horsepower` = 98
        prediction = model.get_prediction([1,98])
        summary_frame = prediction.summary_frame()
        print("\n=========================Summary Frame============================\n")
        print(summary_frame)
        print("\n=========================End of Summary Frame======================\n")
        """
        This will provide the predicted mpg, and the 95% confidence and prediction intervals.
        """

        # Part (b)
        # Step 4: Plot the response and predictor
        plt.scatter(autoData['horsepower'], autoData['mpg'], label='Data points')
        plt.plot(autoData['horsepower'], model.fittedvalues, color='red', label='Regression line')
        plt.xlabel('Horsepower')
        plt.ylabel('MPG')
        plt.title('Horsepower vs MPG')
        plt.legend()
        # plt.show()

        # Part (C)
        # Step 5: Diagnostic plots
        # 1. Residuals vs Fitted values
        sm.graphics.plot_regress_exog(model, 'horsepower', fig=plt.figure(figsize=(12, 8)))
        # plt.show()

        #QQ plot
        sm.qqplot(model.resid, line='s')
        plt.show()

        """
            Comments on the diagnostic plots
                Residuals vs Fitted: Look for patterns. A random scatter suggests a good fit.
                QQ Plot: Check if residuals follow a normal distribution. Points should lie on the line.
                This approach will help you perform and interpret a simple linear regression analysis using the Auto dataset.
        """

RegressionAnalysis.simpleLinearRegressionOnAutoDataset()