for the past week I have made immense progress
- cleaned data 
- made eda notebook for plots
- made csv for tableau


cdc_weekly_clean
- fields
- types 
- units (per 100k)
- filters applied (US, sex=overall)
cdc_disparities_by_week
- how gaps/ratios are computed 
- reference = NH_White 
brfss2020_model_ready
- target = HeartDisease
brfss2022_model_ready
- target = HadHeartAttack
equity_selected
- columns kept - DiagperiodL90D timeliness prox

1. COVID-19 death rates over time by race (cdc weekly data)
- tracked each groups  on severety and persistent impact of pandemic
- CDC reported end of every epidemiological week data by grouped by race_h
- peaks aligning with major COVID waves like delta and omnicron
- disparities narrow but remaiin persistent
- real world health inequality, showing racial disparities in healthcare outcomes, that existed well before AI involvement 

2. Heart Disease Prevalence by Race BRFSS 2020
- bar chart comparing mean heart disease rates across racial categories in2020 behavioral risk factor survelleiance system
- by groupong RaceH and the mean of heartdisease (made into binary)
- this plot links lifestyle and chronic disease to racial health inequalities
- giving a baseline measure of chronic-disease disparity before any modeling

3. Distribution of BMI
- histogram with kernel-density overlay showing body mass index distribution across 2022 BRFSS 
- most participants being between 24-30
- right-skew highlights rising obesity but smaller subset 
- indicating population level health patterns 
- centered 27-28 indicating that overwight status is the norm among adults
- highlighting population health trends relevent to equity in disease prevention 

4. ROC BRFSS 2020
- ROC curve plotting the true positive rate (TPR) vs the false positive rate (FPR) or logisitc regression predicting heart disease 
- TPR - correctly identified heart disease 
  - tp/(tp+fn)
  - high tpr = catches most real positives 
  - low tpr = misses many real cases 
- FPR - healthy people incorrectly flagged positive
  - fp/(fp+tn)
  - low = fewer false alarms 
  - high = more unnecessary alerts or treatments 
- AUC ≈ 0.84 indicating strong predictive ability, if it was 1.0 it would be a perfect model 
  - auc = roc_auc_score(y_test, y_proba)
  - y_proba = models predicted probabilities of "heart disease"
- curve well above diagonal line
- relatively low false positive rate 
- cann detect heart disease risk patterns accurately enough to study fairness accross groups
- model correctly identifies over 80% of true heart disease cases while only falsely flagging about 30% of healthy individuals 

5. per-race metrics (test)
- 2020 BRFSS 
- grouped bar chart comparing the AUC, ACC, and F1 scores accross racial groups 
- f1 - (2*precision *recall)/(precision + recall)
  - precision is how often the AI makes a correct diagnoses
  - recall is how many of the people that were diagnosed did AI catch 
  - compared metrics per racial subgroups 
  - highest being Asians, lowest being AIAN
  - F1 lowest for Hispanic and asian groups 
  - although overall accurate model, it performs unevenly depending on race, an indicator of algorithmic bias

6. Death rate ratio vs nh_white (50-64y)
- each line representing how much higher or lower another racial group's death rate was compared to nh/white gorup ages 50-64
- the dashed line at 1.0 marks parity, anything above means that the groups death rate was higher than the white baseline 
- death rate ratio = death rate of group / death rate of nH_white then filtered for age 50-64
- early 2020 
  - AIAN death rate peaked 15-20x more than nh_white
  - hispanic ~10x higher
  - NH_black 4-6x higher
  - Asian near or below parity
- 2021-23
  - ratios shrink but remain for AIAN, NH_Black, and Hispanic
- source of bias health outcomes are not evenly distributed, even before the model
