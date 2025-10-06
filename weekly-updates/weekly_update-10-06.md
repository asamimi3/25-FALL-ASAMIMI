AI in Healthcare: Equity or Inequity?

Worked on environment setup.
Implemented automated directory creation.

Created a cleaning framework in order to 
- harmonize race and ethnicity categories with `RACE_MAP`  
- change yes/no responses to binary format
- winsorized numeric columns to reduce outliers 
- saved intermediate outputs in parquet format for SQL and Tableau later on

For CDC data set built a function `clean_cdc_weekly()`
- Load weekly demographic COVID-19 data from CDC
- normalized column names 
- filtered data for US only and sex = "Overall"
- map race/ethnicity using `RACE_MAP`
- tidy dataset with key metrics 
- disparities vs. non-hispanic White population by week and age
- saved parquet

in progress for completion this coming week
BRFSS 2020 Heart Disease Data 
- intiated function `clean_brfss_2020()`
- change yes/no responses to binary 
- ordinal health scales numerically 
- convert and winsorize numeric fields for `BMI`, `PhysicalHealth`, etc
- harmonize race and age for consistency with CDC data set 
- prep data for model training and fairness evaluation


