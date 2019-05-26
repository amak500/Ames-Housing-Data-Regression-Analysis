# Project 1: Ames Housing Data Analysis

## Problem Statement: 
 How do we draw people to the Central Iowa MLS and increase revenue flow from users? Currently we are facing competition from 700 other MLS services in the country, let alone the 18 that reside in Iowa. Additionally, we have to compete with online services such as Trulia, Redfin, and Zillow, which all have housing price algorithms that help people decided just how much a house is truly worth. I propose that we use ridge regression modeling in order to create our own housing price algorithm, so that we can stay competitive in the modern marketplace and give ourselves a leg up against the other MLS competition. 
  
## Data dictionary
    
Data Dictionary and Feature dtypes

|Feature | Type | Cat/Cont | Description|
| --- | --- | --- | --- |
SalePrice | int | continuous | The property's sale price in dollars. This is the target variable.
MSSubClass | int | categorical | The building class.
MSZoning | object | categorical |  Identifies the general zoning classification of the sale.
LotFrontage | float | continuous | Linear feet of street connected to property.
LotArea | int | continuous | Lot size in square feet.
Street | object | categorical | Type of road access to property.
Alley | object | categorical | Type of alley access to property.
LotShape | object | categorical | General shape of property.
LandContour | object | categorical | Flatness of the property.
Utilities | object | categorical | Type of utilities available.
LotConfig | object | categorical | Lot configuration.
LandSlope | object | categorical | Slope of property.
Neighborhood | object | categorical | Physical locations within Ames city limits.
Condition1 | object | categorical | Proximity to main road or railroad.
Condition2 | object | categorical | Proximity to main road or railroad (if a second is present).
BldgType | object | categorical | Type of dwelling.
HouseStyle | object | categorical | Style of dwelling.
OverallQual | int | ordinal | Overall material and finish quality.
OverallCond | int | ordinal | Overall condition rating.
YearBuilt | int | continuous | Original construction date
YearRemodAdd | int | continuous | Remodel date (same as construction date if no remodeling or additions).
RoofStyle | object | categorical | Type of roof.
RoofMatl | object | categorical | Roof material.
Exterior1st |object | categorical |  Exterior covering on house.
Exterior2nd |object | categorical | Exterior covering on house (if more than one material).
MasVnrType | object | categorical | Masonry veneer type.
MasVnrArea | float | continuous | Masonry veneer area in square feet.
ExterQual | object | categorical | Exterior material quality.
ExterCond | object | categorical | Present condition of the material on the exterior.
Foundation | object | categorical | Type of foundation.
BsmtQual | object | categorical | Height of the basement.
BsmtCond | object | categorical | General condition of the basement.
BsmtExposure | object | categorical | Walkout or garden level basement walls.
BsmtFinType1 | object | categorical | Quality of basement finished area.
BsmtFinSF1 | float| continuous | Type 1 finished square feet.
BsmtFinType2 | object | categorical | Quality of second finished area (if present).
BsmtFinSF2 | float| continuous | Type 2 finished square feet.
BsmtUnfSF | float| continuous | Unfinished square feet of basement area.
TotalBsmtSF | float| continuous | Total square feet of basement area.
Heating | object | categorical | Type of heating.
HeatingQC | object | categorical | Heating quality and condition.
CentralAir | object | categorical | Central air conditioning.
Electrical | object | categorical | Electrical system.
1stFlrSF | int | continues | First Floor square feet.
2ndFlrSF | int | continues | Second floor square feet.
LowQualFinSF | int | continues | Low quality finished square feet (all floors).
GrLivArea | int | continues | Above grade (ground) living area square feet.
BsmtFullBath | float| continuous | Basement full bathrooms.
BsmtHalfBath | float| continuous | Basement half bathrooms.
FullBath | int | continues | Full bathrooms above grade..
HalfBath | int | continues | Half baths above grade.
Bedroom | int | continues | Number of bedrooms above basement level.
Kitchen | int | continues | Number of kitchens.
KitchenQual | object | categorical | Kitchen quality.
TotRmsAbvGrd | int | continues | Total rooms above grade (does not include bathrooms).
Functional | object | categorical | Home functionality rating.
Fireplaces | int | continues | Number of fireplaces.
FireplaceQu | object | categorical | Fireplace quality.
GarageType | object | categorical | Garage location.
GarageYrBlt | float | continues | Year garage was built.
GarageFinish | object | categorical | Interior finish of the garage.
GarageCars | float | continues | Size of garage in car capacity.
GarageArea | float | continues | Size of garage in square feet.
GarageQual | object | categorical | Garage quality.
GarageCond | object | categorical | Garage condition.
PavedDrive | object | categorical | Paved driveway.
WoodDeckSF | int | continues | Wood deck area in square feet.
OpenPorchSF | int | continues | Open porch area in square feet.
EnclosedPorch | int | continues | Enclosed porch area in square feet.
3SsnPorch | int | continues | Three season porch area in square feet.
ScreenPorch | int | continues | Screen porch area in square feet.
PoolArea | int | continuous | Pool area in square feet.
PoolQC | object | categorical | Pool quality.
Fence | object | categorical | Fence quality.
MiscFeature | object | categorical | Miscellaneous feature not covered in other categories.
MiscVal | int | continuous | $Value of miscellaneous feature.
MoSold | int | continuous | Month sold.
YrSold | int | continuous | Year sold.
SaleType | object | categorical | Type of sale.|

## Data Cleaning

I loaded in the Ames training data, and appropriately replaced the null values in the dataset to signify the lack of feature. I then converted the names of categories within certain features such as MS SubClass and MS Zoning in order to make them more human readable. I then converted Quality and Condition features in object types by assigning the numbers 1-3 to bad, 4-7 to good, and 8-10 to excellent, since I did not believe that quality and condition have a linear relationship with sale price. Finally, I constructed plots of the cleaned data and analyzed them. It was found that housing prices were skewed to the right by some outliers in the 600,000 range. Additionally, Northridge and Northridge Heights had the highest average housing prices when compared to other neighborhoods. As expected, houses with Excellent quality had the highest average housing prices when compared to Good and Bad houses, even having their average be roughly equal to the 75th percentile of housing prices of Good houses. Finally, properties zoned to be Floating Village Residential communities had the highest average housing prices when compared to other types of zoned properties, which makes sense because they are general elderly retirement communities. After I finished my data analysis, I exported the cleaned data to be preprocessed and dummied out. 

## Preprocessing and Dummying

I loaded in the cleaned data from the previous section, dropped the PID column as it was an unnecessary categorical variable, and then dummied out the remaining categorical variables. This increased the number of features I was predicting on from 80 to 377. Additionally, I used sklearn's VarianceThreshhold function in order to remove all features with a variance less than 0.05, and sklearn's StandardScaler Function to scale my features to be normally distributed, so that there wouldn't be any issues related to features being on different scales from one another.   

## Modeling

During this project I constructed 4 models in total: the naive model, a multiple linear regression model, a lasso regularization model, and a ridge regularization model. The root mean square errors of the models were respectively 78375.4,16879901904491.34, 24723.187738745542, and 24601.381265834087. Thus, my multiple linear regression model did worse than my lasso naive model, while my lasso regularization model and ridge regularization model did better. Additionally, since my ridge regularization model did better than my lasso regularization model, I am choosing it to be my production model. According to my ridge regression model, the most important features according to this model for predicting sale prices are all related to the quality of a house, be it overall, or of various features such as first floor square footage or the kitchen. The least important features seemed to be tied to when the property was sold and the property's exterior. 
## Model Improvement

I experimented with changing the range of alpha value to search for, but to not much success. 

##  Conclusion
We should use our ridge regression model in order to convince more realtors to sign up for our MLS service, as we can increase their revenues by allowing them to specifically target the best homes to try to sell and avoid the ones that aren't worth the effort. Additionally, we would urge realtors to try to sell homes that have good overall quality and large lot area, and to avoid homes that have bad fence and metal siding quality. We should also tell realtors to invest in the Northridge, Northridge Heights, and Stonebrook neighborhoods, as they have the highest average home prices from what we've seen. They should also try to avoid Meadow Village, Brookside, and the Iowa DOT and Rail Road neighborhoods, as they have the lowest average home prices. 

In order to generalize my model to other cities outside the state of Iowa, such as Los Angeles or New York, I would need to compile more data about the area's features, and possibly switch to logistic regression to ddeal with the much larger variance in home sale prices between regions. 