"# wine-types-and-quality-prediction" 

Data Source: https://archive.ics.uci.edu/ml/datasets/wine+quality

classes:

if quality<=5: "bad"

if quality==6: "normal"

if quality>6: "good"



Results on test set:

1. Random Forest Classifier:

    accuracy_score:  0.7309782608695652

    recall_score:  0.7443390216251573

    precision_score:  0.7110573585074693


2. Gradient Boost Classifier:

    accuracy_score:  0.6983695652173914

    recall_score:  0.6985130981021118

    precision_score:  0.6922148312392215


3. Adaboost Classifier:

    accuracy_score:  0.7309782608695652

    recall_score:  0.7463920817369093

    precision_score:  0.7246366100024636

  

