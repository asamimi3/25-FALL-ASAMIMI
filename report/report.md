draft 
Does AI in Healthcare reduce or exacerbate disparities?
A Data-Driven Analysis Using BRFSS and CDC COVID-19 Surveillance Data

Intro
As a long-standing feature of U.S. healthcare, racial disparities in health outcomes has been a part of the industry
for a long time. Communites of color face disproportionate burdens in chronic disease, mortality, and access to care. 
As healthcare increasingly adopts AI into te decision-making and predictive modeling process, it raises the question:
Does AI reduce inequalities or does it amplify them?
To be able to properly answer this, I have conducted an in deoth analysis combining:
1. Epidemic data from CDC  COVID-19 weekly surveillance dataset.
2. Chronic disease and risk factor data from the BRFSS 2020-2022 surveys
3. A predictive machine learning model trained on BRFSS 2020
4. A fairness audit examining AI performance across racial groups
5. Tableau dashboards to visualize disparities and model behavior
This report integrates public health data and AI outcomes in order to evaluate if predictive models behave equitably 
and how existing disparities influence the algorithm. 

Data Sources 
CDC COVID-19 Weekly Surveillance Data (2020-2023)
This data included columns like, `End of Week`, `Age Group`, `Race/Ethnicity`, `Case Rate`, `Death Rate`, that provided 
weekly COVID-19 mortality rate per 100k and was used to evaluate disparities in acute health outcomes per race with 
seperate extraction for all adults in compared to a high-risk group 50-64 year olds. Which provided insight into 
mortality curves consistently showing large racial gaps early in the pandemic. 

BRFSS 2020 & 2022 (Behavioral Risk Factor Surveillance System)
This data included variables like `Race/Ethnicity`, `Heart disease diagnosis`, `BMI`, `Chronic disease risk factors`, 
that help provide insight into the chronic disease burdens that differ strongly across racial groups and underlie later 
disparities in racial inequality. 

Methods
From the CDC COVID-19 data, it unveiled
- every racial minority group had higher cumulative COVID-19 mortality than 
NH/White.
- NH/Black and AIAN populations experienced the highest peaks 
- Mortality decreased sharply for all groups after 2022, but disparities persisted

From the BRFSS data, it unveiled
- Heart disease prevalence disproportionately affected `NH/Black`, `AIAN`, and `NH/White`
- `Asian` respondents consistently had the lowest recorded heart disease and showed lowest BMI
- BMI distributions showed higher obesity prevalence among `NH/Black` and `AIAN` populations
These patterns provide context and indicate broader structural health inequalities. 

Predictive Modeling
Models Tested 
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
Each model predicted heart disease 0 or 1.

Train/Test Split
I split the BRFSS data, 80% of which was for training the model, then the other 20% was for testing the model. The data 
was stratified to maintain class balance. Then GridSearchCV hyperparameter tuning performed for all models.
The best Model was logistic regression, that came up with an AUC of 0.84, Accuracy of 0.75, and an F1 score of 0.34.
While the Gradient Boosting had the highest AUC of 0.84, it had the lowest F1 score. Lastly Random Forest achieved high 
accuracy but a poor F1, this is due to the imbalance across racial groups. 

Fairness Analysis Across Racial groups 
Fairness was evaluated per group:
Metrics calculated 
- AUC
- ACC
- F1 score
- TPR / FPR 

Fairness Findings 
- Model accuracy was similar across all race groups (0.82 - 0.88)
- AUC remained stable (0.83 - 0.87) across groups 
- F1 scores were uniformly low because heart disease is rare 
- No racial group was disproportionately misclassified 
- ROC curves by race show no major divergence 

Showing that the predictive model did not introduce any new disparities and performed comparably across racial groups.

Results 
Dashboard 1 - real world health inequity
visuals including:
1. CDC COVID Death Rates by Race (All Adults)
2. CDC COVID Death Rates by Race (50-64 years)
3. BMI Distribution
4. Average Heart Disease by Race (BRFSS 2020)

From the visuals we can see the racial disparities shown before the implementation of AI, this reflects 
in both data sets BRFSS and CDC COVID. Also demonstrating the environment in which AI systems operate.

Dashboard 2 - Predictive Model & Fairness 
Visuals include:
1. ROC Curve - Logistic Regression with random guessing diagonal baseline 
2. Model Comparison of Logistic Regression, Gradient Boost, and Random Forest AUC, Accuracy, and F1)
3. Fairness Across Race with AUC, Accuracy, and F1 per race. 

Key findings from the predictive models we see the AI model itself is fair, showing no major cross-group performance
disparities.


Discussion
The data shows that COVID-19 mortality disparities were large and clear, as well as chronic disease and BMI disparities 
across races. These structural inequalities shape risk landscape in healthcare. However, the AI predictive model did NOT
Replicate or amplify those disparities. 
The predictive modelign actually shows that the AI was fair here. The Logistic Regression is simple and transparent. 
The test data, a part of BRFSS CDC data was not proxied for race, the model used balanced class weights which reduce
the bias toward the majority classes and with the proper evaluation AUC, Accuracy, and F1, per-race metrics, fairness 
was confirmed.
The limitations of BRFSS, is that it is self reporting data and not clinically verified, with real clinical data the
model may behave differently. 

Conclusion 
THis project supports a nuanced conclusion, AI in healthcare does not inherently worsen or reduce disparities, but its 
fairness depends on careful ethics, design, balanced data across races, and transparent evaluation. Although my model 
showed to be fair, pre-existing structural health inequities without AI remain severe. The CDC COVID-19 mortality 
disparities were very large, that is real data that still may be missing thousands of unaccounted for lives, due to the 
existing disparities in todays healthcare system. The AI fairness alone does not fix the systemic inequalities, but it
could certainly be a start. AI can be a huge part of the solution, but only with ethical guardrails, representative
data, and continuous fairness monitoring. 

