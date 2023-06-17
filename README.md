# Predicting Horse Colic Survival: A Machine Learning Approach for Early Intervention
## 6/7/23 
## X23 Group 15 Members:
- Andrew Kassis
- Christian Vargas
- Deniz Erisgen
- Tiffany Andersen

## Introduction
This data science project leverages historical medical records to effectively determine the survival probability of horses with colic by utilizing predictive models that evaluate the likelihood of their survival, taking into account past medical conditions. This project was undertaken because colic poses a significant threat to horses, resulting in a high number of deaths and often being challenging to detect (Egenvall et al., 2008). Colic leads to an estimated annual cost of $115.3 million to the US equine community (Traub-Dargatz et al., 2001). Additionally, horse owners consider colic a major concern (Mellor et al., 2001).

Therefore, the hypothesis being tested is whether we can predict the survivability of a horse with colic based on the features observed during a veterinary visit. The objective and purpose of this research are to develop a predictive model capable of estimating the probability of survival for horses affected by colic. This project aims to provide valuable insights to veterinarians and horse owners, enabling them to make well-informed decisions regarding treatment options and care. Furthermore, the objective involves conducting an in-depth analysis to identify the most influential features or attributes for accurately predicting colic occurrences.

## Dataset
- Dataset can be found on [Kaggle](https://www.kaggle.com/datasets/uciml/horse-colic) or at the original data source over, the [ UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/47/horse+colic)
- Dataset includes 299 instances for training and 89 instances for testing, 388 instances in total.
- Dataset feature types are 17 objects , 7 floats , 4 ints in total 28. 


<details>
 <summary> Attribute Information</summary>

     1:  surgery?
     - 1 = Yes, it had surgery
     - 2 = It was treated without surgery

     2:  Age 
     - 1 = Adult horse
     - 2 = Young (< 6 months)

     3:  Hospital Number (dropped)
     - numeric id
     - the case number assigned to the horse (may not be unique if the horse is treated > 1 time)

     4:  rectal temperature
     - linear
     - in degrees celsius.
     - An elevated temp may occur due to infection.
     - temperature may be reduced when the animal is in late shock
     - normal temp is 37.8
     - this parameter will usually change as the problem progresses eg. may start out normal, then become elevated because of the lesion, passing back through the normal range as the horse goes into shock

     5:  pulse 
     - linear
     - the heart rate in beats per minute
     - is a reflection of the heart condition: 30 -40 is normal for adults
     - rare to have a lower than normal rate although athletic horses may have a rate of 20-25
     - animals with painful lesions or suffering from circulatory shock may have an elevated heart rate

     6:  respiratory rate
     - linear
     - normal rate is 8 to 10
     - usefulness is doubtful due to the great fluctuations

     7:  temperature of extremities
     - a subjective indication of peripheral circulation
     - possible values:
          - 1 = Normal
          - 2 = Warm
          - 3 = Cool
          - 4 = Cold
     - cool to cold extremities indicate possible shock
     - hot extremities should correlate with an elevated rectal temp.

     8:  peripheral pulse
     - subjective
     - possible values are:
          -  1 = normal
          -  2 = increased
          -  3 = reduced
          -  4 = absent
     - normal or increased p.p. are indicative of adequate circulation
               while reduced or absent indicate poor perfusion

     9:  mucous membranes
     - a subjective measurement of colour
     - possible values are:
          - 1 = normal pink
          - 2 = bright pink
          - 3 = pale pink
          - 4 = pale cyanotic
          - 5 = bright red / injected
          - 6 = dark cyanotic
     - 1 and 2 probably indicate a normal or slightly increased circulation
     - 3 may occur in early shock
     - 4 and 6 are indicative of serious circulatory compromise
     - 5 is more indicative of a septicemia

     10: capillary refill time
     - a clinical judgement. The longer the refill, the poorer the
               circulation
     - possible values
          -  1 = < 3 seconds
          -  2 = >= 3 seconds

     11: pain - a subjective judgement of the horse's pain level
     - possible values:
          - 1 = alert, no pain
          - 2 = depressed
          - 3 = intermittent mild pain
          - 4 = intermittent severe pain
          - 5 = continuous severe pain
     - should NOT be treated as a ordered or discrete variable!
     - In general, the more painful, the more likely it is to require surgery
     - prior treatment of pain may mask the pain level to some extent

     12: peristalsis                              
     - an indication of the activity in the horse's gut. As the gut
               becomes more distended or the horse becomes more toxic, the
               activity decreases
     - possible values:
          - 1 = hypermotile
          - 2 = normal
          - 3 = hypomotile
          - 4 = absent

     13: abdominal distension
     - An IMPORTANT parameter.
     - possible values
          - 1 = none
          - 2 = slight
          - 3 = moderate
          - 4 = severe
     - an animal with abdominal distension is likely to be painful and
               have reduced gut motility.
     - a horse with severe abdominal distension is likely to require
               surgery just tio relieve the pressure

     14: nasogastric tube
     - this refers to any gas coming out of the tube
     - possible values:
          - 1 = none
          - 2 = slight
          - 3 = significant
     - a large gas cap in the stomach is likely to give the horse
               discomfort

     15: nasogastric reflux
     - possible values
          - 1 = none
          - 2 = > 1 liter
          - 3 = < 1 liter
     - the greater amount of reflux, the more likelihood that there is some serious obstruction to the fluid passage from the rest of the intestine

     16: nasogastric reflux PH
     - linear
     - scale is from 0 to 14 with 7 being neutral
     - normal values are in the 3 to 4 range

     17: rectal examination - feces
     - possible values
          - 1 = normal
          - 2 = increased
          - 3 = decreased
          - 4 = absent
     - absent feces probably indicates an obstruction

     18: abdomen
     - possible values
          - 1 = normal
          - 2 = other
          - 3 = firm feces in the large intestine
          - 4 = distended small intestine
          - 5 = distended large intestine
     - 3 is probably an obstruction caused by a mechanical impaction
               and is normally treated medically
     - 4 and 5 indicate a surgical lesion

     19: packed cell volume
     - linear
     - the # of red cells by volume in the blood
     - normal range is 30 to 50. The level rises as the circulation
               becomes compromised or as the animal becomes dehydrated.

     20: total protein
     - linear
     - normal values lie in the 6-7.5 (gms/dL) range
     - the higher the value the greater the dehydration

     21: abdominocentesis appearance
     - a needle is put in the horse's abdomen and fluid is obtained from the abdominal cavity
     - possible values:
          - 1 = clear
          - 2 = cloudy
          - 3 = serosanguinous
     - normal fluid is clear while cloudy or serosanguinous indicates a compromised gut

     22: abdomcentesis total protein
     - linear
     - the higher the level of protein the more likely it is to have a compromised gut. Values are in gms/dL

     23: outcome (TARGET)
     - what eventually happened to the horse?
     - possible values:
          - 1 = lived
          - 2 = died
          - 3 = was euthanized

     24: surgical lesion?
     - retrospectively, was the problem (lesion) surgical?
     - all cases are either operated upon or autopsied so that this value and the lesion type are always known
     - possible values:
          - 1 = Yes
          - 2 = No

     25, 26, 27: type of lesion
     - first number is site of lesion
          - 1 = gastric
          - 2 = sm intestine
          - 3 = lg colon
          - 4 = lg colon and cecum
          - 5 = cecum
          - 6 = transverse colon
          - 7 = retum/descending colon
          - 8 = uterus
          - 9 = bladder
          - 11 = all intestinal sites
          - 00 = none
     - second number is type
          - 1 = simple
          - 2 = strangulation
          - 3 = inflammation
          - 4 = other
     - third number is subtype
          - 1 = mechanical
          - 2 = paralytic
          - 0 = n/a
     - fourth number is specific code
          - 1 = obturation
          - 2 = intrinsic
          - 3 = extrinsic
          - 4 = adynamic
          - 5 = volvulus/torsion
          - 6 = intussuption
          - 7 = thromboembolic
          - 8 = hernia
          - 9 = lipoma/slenic incarceration
          - 10 = displacement
          - 0 = n/a

     28: cp_data
     - is pathology data present for this case?
          - 1 = Yes
          - 2 = No
     - this variable is of no significance since pathology data is not included or collected for these cases

</details>

---

## Feature Engineering
Various feature engineering techniques are employed to prepare the dataset for further analysis, taking into account the percentage of missing values and the data type of each column.

- ***One-Hot Encoding***: Categorical data is converted into numeric data using one-hot encoding.

- ***Removal of columns with high missing percentages***: Columns with more than 50% missing values are considered for removal from the dataset, eliminating columns with insufficient data.

- ***Mode imputation*** for categorical columns: Missing values in categorical columns (identified by the dtype == 'object' condition) are filled using the mode.

- ***Median imputation*** for numerical columns: Missing values in numerical columns are filled using the median value of that specific column.

## Models Compared 
Below you can find the models we used for this data set and their training and test accuracies
| Model| Training Accuracy | Test accuracy |
| -- | --- |- | 
| Random Forest Classifier | 95.65% |95.51% |
|  K-Nearest Neighbors Classifier   | 74.25% | 76% |
|Logistic Regression| 65.89% | 67.42%|
| Decision Trees| 78.6% | 76.4%|

## Methods 
We did not use anything but the dataset from the UCI Machine Learning Repository. We trained our dataset using different models to find the most accurate one for predicting if a horse will survive. 

The tools, packages, and libraries used include:
- numpy
- pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- Graphviz
- Google Colab
- GitHub
- Kaggle

## Results 
After evaluating multiple models, we found that the Random Forest Classifier (RFC) outperformed the other four models in terms of accuracy for predicting horse survival. By utilizing RFC, we were able to achieve an impressive prediction accuracy of 95%.

## Discussion

The information from this study implies that there are certainly characteristics that correlate with the survivability of colic in horses. This is important because, as stated above, colic often leads to death in horses, and caretakers, including veterinarians, are often unsure about the indicators to look for. The perspective for future research would involve improved feature engineering, considering that this study really only utilizes One-Hot Encoding. There could be potential for combinations of features, such as the three lesion columns, which could help in more accurately predicting the diagnosis and treatment of colic in horses. All three lesion columns appeared on the top ten correlation values to 'outcome'. There also needs to be more analysis on the total_protein column, but due to missing values this column proved to only yield a correlation value of 0.26. 

Additionally, more advanced machine learning models, like recurrent neural networks, could be implemented. One interesting aspect that future researchers should consider is using external validation by collecting other datasets related to horse colic and performing validation of the newly created models. This could lead to interesting results, considering that the dataset used for this project is from 1998 and there is a need for an updated dataset with more information. In any case, this report outlines certain characteristics in horses with colic that are more prevalent among those that ultimately die. Although the correlation could be stronger, there is certainly something worth investigating further, as it has the potential to benefit the equestrian community in numerous ways.

## Summary

This project employed a diverse range of models, including the Random Forest Classifier, K-Nearest Neighbors Classifier, Logistic Regression, and Decision Tree, to predict the outcome of horses with colic. Among these models, the Random Forest Classifier demonstrated the highest accuracy, achieving a test accuracy rate of over 96 percent. The Decision Tree model followed closely behind, with a test accuracy exceeding 83 percent. While the Random Forest Classifier yielded impressive results in this project, further investigation is recommended. Conducting in-depth analysis, utilizing an updated dataset, and performing external validation could enhance the accuracy and reliability of the predictive models.

Although individual features alone could not reliably predict the horse's outcome, three features showed strong correlations: packed_cell_volume, pulse, and total protein. These attributes played a significant role in determining the prognosis of the horses. Additionally, two noteworthy feature relationships emerged from the analysis: the relationship between peristalsis (gut activity) and abdominal distention (bloating), as well as the association between the absence of feces and the likelihood of requiring surgery.

Given that the dataset is from 1998, the information may not reflect the most current information and practices related to horse colic. Additionally, there could be new variables captured in a newer dataset that could yield higher and more consistent results. Moreover, the sample size was relatively small, making certain machine learning tasks less feasible. Another issue was the 30% of missing values in the dataset, resulting in data quality and incomplete information concerns. Although our data cleaning and preprocessing addressed many of these cases, the analysis and modeling could have been improved with a more complete dataset.

## References

Egenvall, A., Penell, J., Bonnett, B. N., Blix, J., & Pringle, J. (2008). Demographics and costs of colic in Swedish horses. *Journal of veterinary internal medicine, 22*(4), 1029–1037.

McLeish,Mary and Cecile,Matt. (1989). Horse Colic. UCI Machine Learning Repository. https://doi.org/10.24432/C58W23.

Mellor, D. J., Love, S., Walker, R., Gettinby, G., & Reid, S. W. (2001). Sentinel practice-based survey of the management and health of horses in northern Britain. *The Veterinary record, 149*(14), 417–423.

Traub-Dargatz, J. L., Kopral, C. A., Seitzinger, A. H., Garber, L. P., Forde, K., & White, N. A. (2001). Estimate of the national incidence of and operation-level risk factors for colic among horses in the United States, spring 1998 to spring 1999. *Journal of the American Veterinary Medical Association, 219*(1), 67–71.

