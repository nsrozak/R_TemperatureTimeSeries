# Seattle Temperature Forecasting

## Introduction

This project uses R to forecast the monthly average temperature in Seattle.

## File Descriptions

***seattle_data.csv:*** Data file for the project.

***TimeSeriesAnalysis.Rmd:*** Raw Rmarkdown code.

***TimeSeriesAnalysis.md:*** Code for this project in a format that is accessible by GitHub.

## Findings

Seattle temperature data can be decomposed into a seasonal component and a random component. There has been a slight increase in trend of temperature data from 1950 to 2020, but this trend varies greatly from year to year so I did not include this component in my models. To make the data stationary, I Box-cox transformed the data to correct for heteroskedasticity and removed the seasonal component by differencing at lag 12.

I modeled the data with exponential smoothing; SARIMA; and common machine learning algorithms like linear regression, artificial neural networks, and gradient boosted trees. These models were trained on data from 1950 to 1999, and they were tested on data from 2000 to 2020. Grid search was used to tune hyperparameters. The exponential smoothing model produced the best forecasts for 2000, and the best forecasts for 2000 through 2020 were produced by the SARIMA model. 

## Improvements

*Exponential Smoothing*

- Make adjustments to the model so it passes all diagnostics tests

*SARIMA*

- Several models tested through grid search were unable to converge to final parameters. Therefore, these models were not considered in the model selection process. Adjustments to the optimizer could be made to ensure that all potential models are produced and examined.

*Machine Learning*

- I examined the ACF graph to choose which lags to use as predictors for the models. However, a more sophisticated approach could have yielded better predictors. Additionally, rolling means and other types of predictors related to time could have been included in the models. 
- Forward Chaining was used to tune hyperparameters, meaning the data at the beginning of the dataset influenced the RMSE at each fold. Cross validation with sliding windows would give each time segment equal consideration, so this may have been a better method to use when tuning hyperparameters.
- Several articles mentioned specialized machine learning models for time series. For example, there are types of regression models, support vector machines, and neural networks that are designed specifically for time series. However, I need to do further research so I have a solid understanding of these models before I'd feel comfortable implementing them. 
