import pandas as pd
import numpy as np
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.drift_detection.ddm import DDM
from skmultiflow.drift_detection.eddm import EDDM
from skmultiflow.drift_detection.kswin import KSWIN
from skmultiflow.drift_detection.page_hinkley import PageHinkley



neighbors_number = 2
training_test_split_point = 0.2
accuracy_to_print = []
symbols = []

delta_adwin = 0.002
adwin = ADWIN(delta=delta_adwin)
adwin_change = []
adwin_warning = []

ddm_min_num_instances = 30
ddm_warning_level = 2.0
ddm_out_control_level = 3.0
ddm_change = []
ddm_warning = []
ddm = DDM(min_num_instances=ddm_min_num_instances, warning_level=ddm_warning_level,
          out_control_level=ddm_out_control_level)

eddm_change = []
eddm_warning = []
eddm = EDDM()

page_h_change = []
page_h_warning = []
page_h_min_instances = 30
page_h_delta = 0.005
page_h_threshold = 50
page_h_alpha = 0.9999
page_h = PageHinkley(min_instances=page_h_min_instances, delta=page_h_delta,
                     threshold= page_h_threshold, alpha=page_h_alpha)

kswin = KSWIN()
kswin_change = []
kswin_warning = []

pd.set_option('display.max_columns', None)
df = pd.read_csv("train.csv")
# df = df[:10000]
for sym in df['sym']:
    if sym not in symbols:
        symbols.append(sym)

for i, sym in enumerate(df['sym']):
    df.at[i, 'sym'] = symbols.index(sym)

df['is_profit'] = df['is_profit'].astype(float)
y = df["is_profit"].values
tmp_df = df.drop(columns=['is_profit', 'datetime'])
x = tmp_df.values
model = neighbors.KNeighborsClassifier(neighbors_number)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=1-training_test_split_point)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
to_bool_predictions = []
for i in range(len(predictions)):
    to_bool_predictions.append(predictions[i] == y_test[i])
good = 0
accuracy_to_print = []
for i, value in enumerate(to_bool_predictions):
    if value:
        good += 1
    accuracy_to_print.append(good * 100 / (i+1))
for i, prediction in enumerate(to_bool_predictions):
    adwin.add_element(int(prediction))
    ddm.add_element(int(prediction))
    eddm.add_element(int(prediction))
    page_h.add_element(int(prediction))

    kswin.add_element(int(prediction))
    if kswin.detected_change():
        kswin_change.append(i)
    if kswin.detected_warning_zone():
        kswin_warning.append(i)

    if adwin.detected_change():
        adwin_change.append(i)
    if adwin.detected_warning_zone():
        adwin_warning.append(i)
    if ddm.detected_change():
        ddm_change.append(i)
    if ddm.detected_warning_zone():
        ddm_warning.append(i)
    if eddm.detected_change():
        eddm_change.append(i)
    if eddm.detected_warning_zone():
        eddm_warning.append(i)
    if page_h.detected_change():
        page_h_change.append(i)
    if page_h.detected_warning_zone():
        page_h_warning.append(i)

print(f"Adwin w: {adwin_warning}")
print(f"Adwin c: {adwin_change}")
print("-----------")
print(f"DDM w: {ddm_warning}")
print(f"DDM c: {ddm_change}")
print("-----------")
print(f"EDDM w: {eddm_warning}")
print(f"EDDM c: {eddm_change}")
print("-----------")
print(f"PageHinkley w: {page_h_warning}")
print(f"PageHinkley c: {page_h_change}")
print("-----------")
print(f"KSWIN w: {kswin_warning}")
print(f"KSWIN c: {kswin_change}")
print("-----------")

def show_plot(x, y, algorithm, changes, warnings):
    for e in warnings:
        plt.axvline(e, alpha=0.3, color='orange')
    for e in changes:
        plt.axvline(e, alpha=0.3, color='red')

    sns.lineplot(x=x, y=y,
                     alpha=0.4, color='blue')
    plt.title(f'Accuracy of prediction, {algorithm.upper()}')
    plt.xlabel('number of sample')
    plt.ylabel('% of correct predictions')
    plt.tight_layout()

show_plot([x for x in range(len(y_test))], accuracy_to_print, "eddm", eddm_warning, eddm_change)
plt.show()