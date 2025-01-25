import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import csv
import numpy as np
import keras
from keras.callbacks import Callback
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os
import numpy as np
import csv
import tensorflow as tf
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from att import get_model,get_model_noatt,get_model_one_lstm,get_model_nolstm,get_model_no_bp
import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class RocAucCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_true_train = self.model.predict(self.training_data[0])
        y_true_val = self.model.predict(self.validation_data[0])

        roc_auc_train = roc_auc_score(self.training_data[1], y_true_train)
        roc_auc_val = roc_auc_score(self.validation_data[1], y_true_val)

        logs['roc_auc_train'] = roc_auc_train
        logs['roc_auc_val'] = roc_auc_val

        print(f' - roc_auc_train: {roc_auc_train:.4f} - roc_auc_val: {roc_auc_val:.4f}')

class roc_callback(Callback):
    def __init__(self, val_data,name):
        self.mi = val_data[0]
        self.lnc = val_data[1]
        self.y = val_data[2]
        self.name = name

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict([self.mi,self.lnc])
        auc_val = roc_auc_score(self.y, y_pred)
        aupr_val = average_precision_score(self.y, y_pred)
        val_accuracy = logs.get('val_accuracy', 0.0)  
        filename = f"./model/2021bs64/{self.name}Model{epoch}_val_acc_{val_accuracy:.4f}.h5"
        self.model.save_weights(filename)
        # self.model.save_weights(
        #     "./model/2021bs64/%sModel%d.h5" % (self.name, epoch))
        print('\r auc_val: %s ' %str(round(auc_val, 4)), end=100 * ' ' + '\n')
        print('\r aupr_val: %s ' % str(round(aupr_val, 4)), end=100 * ' ' + '\n')
       
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    not_found_counter = 0  

    for i in range(len(InteractionPair)):
        Pair1 = InteractionPair[i][0]
        Pair2 = InteractionPair[i][1]

        found1 = False
        for j in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[j][0]:
                SampleFeature1.append(NodeBehavior[j][1:])
                found1 = True
                break

        if not found1:
            print("Pair not found for Pair1:", Pair1)
            not_found_counter += 1

        found2 = False
        for k in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[k][0]:
                SampleFeature2.append(NodeBehavior[k][1:])
                found2 = True
                break

        if not found2:
            print("Pair not found for Pair2:", Pair2)
            not_found_counter += 1

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    print("Number of pairs not found:", not_found_counter)

    return SampleFeature1, SampleFeature2

def ReadMyCsv1(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  
        SaveList.append(row)
    return


t1 = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

name = 'premiM'
Data_dir='data/'
train = np.load(Data_dir+'train_org.npz')
val=np.load(Data_dir+'val_org.npz')
X_mi_tra, X_M_tra, y_tra = train['X_mi_tra'], train['X_M_tra'], train['y_tra']

X_mi_val,X_M_val, y_val=val['X_mi_val'], val['X_M_val'], val['y_val']

PositiveSample_Train = []
ReadMyCsv1(PositiveSample_Train, Data_dir+'PositiveSample_Train.csv')
PositiveSample_Validation = []
ReadMyCsv1(PositiveSample_Validation, Data_dir+'PositiveSample_Validation.csv')
PositiveSample_Test = []
ReadMyCsv1(PositiveSample_Test, Data_dir+'PositiveSample_Test.csv')

NegativeSample_Train = []
ReadMyCsv1(NegativeSample_Train, Data_dir+'NegativeSample_Train.csv')
NegativeSample_Validation = []
ReadMyCsv1(NegativeSample_Validation, Data_dir+'NegativeSample_Validation.csv')
NegativeSample_Test = []
ReadMyCsv1(NegativeSample_Test, Data_dir+'NegativeSample_Test.csv')

x_train_pair = []
x_train_pair.extend(PositiveSample_Train)
x_train_pair.extend(NegativeSample_Train)
x_validation_pair = []
x_validation_pair.extend(PositiveSample_Validation)
x_validation_pair.extend(NegativeSample_Validation)
AllNodeBehavior = []
ReadMyCsv1(AllNodeBehavior, 'AllNodeBehavior.csv')


x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)

behavior_train_1 = x_train_1_Behavior
behavior_train_2 = x_train_2_Behavior
behavior_val_1 = x_validation_1_Behavior
behavior_val_2 = x_validation_2_Behavior
len_behavior1 = behavior_train_1.shape[1]
len_behavior2 = behavior_train_2.shape[1]
model_dir = 'vic/72'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model = get_model(len_behavior1, len_behavior2)
# model = get_model_none()  

# model = get_model_none()
model.summary()

class SaveEveryEpoch(ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        
        model_path = os.path.join(model_dir, f'model-128__{epoch+1:02d}_val_acc_{val_accuracy:.4f}.h5')
        self.filepath = model_path
        super().on_epoch_end(epoch, logs)


checkpoint_callback = SaveEveryEpoch(
    filepath='',  
    monitor='val_accuracy',
    save_best_only=False,  
    mode='max',
    save_freq='epoch',
    verbose=1
)
# roc_auc_callback = RocAucCallback()
# roc_auc_callback.training_data = ([behavior_train_1, behavior_train_2], y_tra)
# roc_auc_callback.validation_data = ([behavior_val_1, behavior_val_2], y_val)
roc_auc_callback = RocAucCallback()
roc_auc_callback.training_data = ([X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2], y_tra)
roc_auc_callback.validation_data = ([X_mi_val, X_M_val, behavior_val_1, behavior_val_2], y_val)
# roc_auc_callback = RocAucCallback()
# roc_auc_callback.training_data = ([X_mi_tra, X_M_tra], y_tra)  # 更新为只使用 miRNA 和 mRNA 数据
# roc_auc_callback.validation_data = ([X_mi_val, X_M_val], y_val)


#
history = model.fit(
    [X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2],
    y_tra,
    validation_data=([X_mi_val, X_M_val, behavior_val_1, behavior_val_2], y_val),
    epochs=10,
    batch_size=32,
    callbacks=[checkpoint_callback, roc_auc_callback]
)
# history = model.fit(
#     [X_mi_tra, X_M_tra],  
#     y_tra,
#     validation_data=([X_mi_val, X_M_val], y_val),  
#     epochs=10,
#     batch_size=32,
#     callbacks=[checkpoint_callback, roc_auc_callback]
# )

# history = model.fit(
#     [behavior_train_1, behavior_train_2],
#     y_tra,
#     validation_data=([behavior_val_1, behavior_val_2], y_val),
#     epochs=10,
#     batch_size=32,
#     callbacks=[SaveEveryEpoch(filepath='', monitor='val_accuracy', save_best_only=False, mode='max', save_freq='epoch', verbose=1), roc_auc_callback]
# )


latest_model_path = os.path.join(model_dir, f'model_{len(history.epoch):02d}-16_val_acc_{history.history["val_accuracy"][-1]:.4f}.h5')
best_model = keras.models.load_model(latest_model_path)

train_pred = best_model.predict([X_mi_tra, X_M_tra, behavior_train_1, behavior_train_2])
val_pred = best_model.predict([X_mi_val, X_M_val, behavior_val_1, behavior_val_2])

train_auc = roc_auc_score(y_tra, train_pred)
val_auc = roc_auc_score(y_val, val_pred)



# latest_model_path = os.path.join(model_dir, f'model_{len(history.epoch):02d}-16_val_acc_{history.history["val_accuracy"][-1]:.4f}.h5')
# best_model = keras.models.load_model(latest_model_path)
#
# train_pred = best_model.predict([behavior_train_1, behavior_train_2])
# val_pred = best_model.predict([behavior_val_1, behavior_val_2])
#
# train_auc = roc_auc_score(y_tra, train_pred)
# val_auc = roc_auc_score(y_val, val_pred)


# latest_model_path = os.path.join(model_dir, f'model_{len(history.epoch):02d}-16_val_acc_{history.history["val_accuracy"][-1]:.4f}.h5')
# best_model = keras.models.load_model(latest_model_path)
#
# train_pred = best_model.predict([X_mi_tra, X_M_tra])
# val_pred = best_model.predict([X_mi_val, X_M_val])
#
# train_auc = roc_auc_score(y_tra, train_pred)
# val_auc = roc_auc_score(y_val, val_pred)
