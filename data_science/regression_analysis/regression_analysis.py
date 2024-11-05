import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class RegressionAnalysis:
    """
    HOMEWORK 3: PROBLEM 2
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


    def plotPredictStartingSalary():
        # Define the range for GPA (X1)
        GPA = np.linspace(0, 4, 100)

        # Compute salary estimates for both levels (High School and College)
        salary_high_school = 57 + 21*GPA - 10*(GPA*0)
        salary_college = 57 + 21*GPA + 35 - 10*(GPA*1)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(GPA, salary_high_school, label='High School', color='blue')
        plt.plot(GPA, salary_college, label='College', color='green')
        plt.xlabel('GPA (X1)')
        plt.ylabel('Estimated Starting Salary ($1000s)')
        plt.title('Estimated Starting Salary After Graduation')
        plt.legend()
        plt.grid(True)
        plt.show()


"""
HOMEWORK3: EXERCISE 3
    This question involves the use of multiple linear regression on Auto data set.
        (a) Produce a scatterplot matrix which includes all of the variables in the data set.
        (b) Compute the matrix of correlation between the variables using the DataFrame.corr() method.
        (c) Use the sm.OLS() function to perform a multiple linear regression with mpg as the response and all other variables except name as the predictors. Use the summarize() function to print the results. Comment on the output. For instance:
            i. Is there a relationship between the predictors and the response? Use the anovalm() function from statmodels to answer this question.
            ii. Which predictors appear to have a statistically significant relationship to the response?
            iii. What does the coefficient for the year variable suggest?
        (d) Produce some of diagnostic plots of the linear regression fit as described in the lab. Comment on any problems you see with the fit. Do the residual plots suggest any unusually large outliers? Does the leverage plot identify any observations with unusually high leverage?
        (e) Fit some models with interactions as described in the lab. Do any interactions appear to be statistically significant?
        (f) Try a few different transformations of the variables, such as log(X),√X,X2. Comment on your findings.
"""
class MultipleLinearRegressionOnAutoDataset:

    def __init__(self):
        # Load the auto dataset
        # Load the Auto dataset (assuming it is a CSV file)
        # Ref dataset: https://github.com/selva86/datasets/blob/master/Auto.csv
        self.autoData = pd.read_csv('data/auto/Auto.csv')
        self.model = None

    # (a) Product a scatterplot matrix
    def produceScatterplotMatrix(self):
        # Create a scatterplot matrix
        sns.pairplot(self.autoData)
        plt.show()

    # (b)
    def computeTheMatrixOfCorrelation(self):
        # Compute the correlation matrix
        # print(self.autoData)
        correlation_matrix = self.autoData.corr()
        print("=====Correlation Matrix============")
        print(correlation_matrix)
        print("=====End of Correlation Matrix============")

   # (c)
    def performMultipleLinearRegression(self):
       # Define the response and predictors
        X = self.autoData.drop(columns=['mpg', 'name'])
        y = self.autoData['mpg']

        # Add a constant to the predictors
        X = sm.add_constant(X)

        # Fit the model
        self.model = sm.OLS(y, X).fit()

        print("Model param: ", self.model.params[0])

        # Print the summary
        print(self.model.summary()) 

        self.relationshipBetweenPredictorsAndResponse()

    # (c) i relationship between predictors and response
    def relationshipBetweenPredictorsAndResponse(self):
        # Perform ANOVA
        # print("again with summary", self.model.summary())
        anova_results = sm.stats.anova_lm(self.model) #typ=2
        print("=========ANOVA RESULT=================")
        print(anova_results)
        print("=========END ANOVA RESULT=================")

    # (d)
    def diagnosticPlots(self):
        # Residual plots
        plt.figure(figsize=(12, 8))

        # Residuals vs Fitted
        plt.subplot(2, 2, 1)
        plt.scatter(self.model.fittedvalues, self.model.resid)
        plt.xlabel('Fitted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Fitted')

        # Normal Q-Q
        # plt.subplot(2, 2, 2)
        # sm.qqplot(self.model.resid, line='45', ax=plt.gca())
        # plt.title('Normal Q-Q')

        # Scale-Location
        # plt.subplot(2, 2, 3)
        # plt.scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
        # plt.xlabel('Fitted values')
        # plt.ylabel('Sqrt(|residuals|)')
        # plt.title('Scale-Location')

        # Leverage plot
        # plt.subplot(2, 2, 4)
        # sm.graphics.influence_plot(model, ax=plt.gca(), criterion="cooks")
        # plt.title('Leverage vs Residuals')

        # plt.tight_layout()
        # plt.show()



# RegressionAnalysis.simpleLinearRegressionOnAutoDataset()
mr = MultipleLinearRegressionOnAutoDataset()
# mr.produceScatterplotMatrix()
# mr.computeTheMatrixOfCorrelation()
mr.performMultipleLinearRegression()
# mr.diagnosticPlots()