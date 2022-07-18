
import Data.data_provider as dp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

dataset = "fashionMnist"
features, target = dp.load_data(dataset)
x_train, x_test_all, y_train, y_test_all = train_test_split(features, target, test_size=0.4, shuffle=True, random_state=1)
x_test, x_calib, y_test, y_calib = train_test_split(x_test_all, y_test_all, test_size=0.5, shuffle=True, random_state=1) 


# train normal model
model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=1) # 
model.fit(x_train, y_train)
print(model.score(x_test, y_test))

model.fit(x_calib, y_calib)
print(model.score(x_test, y_test))

model2 = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=1) # 
model2.fit(x_calib, y_calib)
print(model2.score(x_test, y_test))
