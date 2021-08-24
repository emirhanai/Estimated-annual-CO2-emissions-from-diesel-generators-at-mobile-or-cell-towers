from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import pandas as pd


def save_decision_trees_as_dot(clf, iteration, feature_name, target_name):
    file_name = open("emirhan_project" + str(iteration) + ".dot",'w')
    dot_data = export_graphviz(
        clf,
        out_file=file_name,
        feature_names=feature_name,
        class_names=target_name,
        rounded=True,
        proportion=False,
        precision=2,
        filled=True,)
    file_name.close()
    print("Decision Tree in forest :) {} saved as dot file".format(iteration + 1))


df = pd.read_csv('co2_country_years_cell_tower_data.csv')

X= df.drop(['Country'], axis = 'columns')
#print(X)
y= df.drop(['2014','2020','2020-Population'], axis= 'columns')
#print(y)

y_data = LabelEncoder()
#LabelEncoder() function :))

y['Country_Data'] = y_data.fit_transform(y['Country'])
# Country Columns value change to Country_Data with fit_transform function

#print(connects)

y_n = y.drop(['Country'],axis='columns')
#New Columns of Target :))

# In additionnn: print(y_n)


feature_names = X.columns
#a few fetaure names..

target_names = y_n.columns
# one of the columns is target name :)

model = RandomForestClassifier(n_estimators=5)

# our model like to above :)

model.fit(X,y_n)
#our model training to the above...

#print(model.estimators_[0])

#The collection of fitted sub-estimators = estimators_


#for i in range(len(model.estimators_)):
    #save_decision_trees_as_dot(model.estimators_[i], i, feature_names, target_names)
    #print(i)


#prediction is the Country!

predict_2014 = input("Estimated annual CO2 emissions (in kilotons) from diesel generators in mobile towers, predict for 2014: ")
predict_2020 = input("Estimated annual CO2 emissions (in kilotons) from diesel generators in mobile towers, predict for 2020: ")
predict_population = input("Enter population: ")

try:
    while True:
        model_run = model.predict([[int(predict_2014),int(predict_2020),int(predict_population)]])
        countrys = pd.read_csv('country_new_arrangement_data.csv',index_col=None, na_values=None)
        countrys_detect_algorithm = countrys.columns.values[model_run]
        print("Predicted country: {}".format(countrys_detect_algorithm))
        break

except:
    print("Try again!")
#print(model.predict([[112,300,200000000]]))

