# =================
# imports
# =================
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.xception import preprocess_input
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image


img_dir = os.path.join('data', 'Garbage_classification')

data = []
for r, d, f in os.walk(img_dir):
    for file in f:
        data.append({'dir': r, 'file': file, 'path': os.path.join(r, file)})
        
data = pd.DataFrame(data)
data['label'] = data['dir'].apply(os.path.split).str[-1]

def load_data(file):
    data = pd.read_csv(file, sep=' ', header=None)
    data.columns = ['file', 'class']
    data['file'] = data['file'].str.replace('.npy', '.jpg')
    data['class'] = (data['class'] - 1).astype(str)
    return data

train_data = data.merge(load_data('data/indexed-files-train.txt'), on=['file'], how='inner').sample(frac=1)
val_data = data.merge(load_data('data/indexed-files-val.txt'), on=['file'], how='inner').sample(frac=1)
test_data = data.merge(load_data('data/indexed-files-test.txt'), on=['file'], how='inner').sample(frac=1)

num_classes = train_data['class'].nunique()
classes = train_data[['class', 'label']].drop_duplicates().sort_values('class')['label'].tolist()
print(classes)

img_height, img_width = 384, 512
res_img_height, res_img_width = 299, 299 

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
)

train_ds = datagen_train.flow_from_dataframe(
    train_data,  
    x_col='path', 
    y_col='class',
    class_mode='categorical',
    target_size=(img_height, img_width),
    color_mode='rgb',
    batch_size=32, 
    shuffle=True,
    save_format='jpg',
)


datagen_eval = tf.keras.preprocessing.image.ImageDataGenerator()

val_ds = datagen_eval.flow_from_dataframe(
    val_data,  
    x_col='path', 
    y_col='class',
    class_mode='categorical',
    target_size=(img_height, img_width),
    color_mode='rgb',
    batch_size=32, 
    shuffle=False,
    save_format='jpg',
)


test_ds = datagen_eval.flow_from_dataframe(
    test_data,  
    x_col='path', 
    y_col='class',
    class_mode='categorical',
    target_size=(img_height, img_width),
    color_mode='rgb',
    batch_size=32, 
    shuffle=False,
    save_format='jpg',
)

# =========================
# data augmentation 
# ==========================

# def prepare_jpg(df_data):
#     files = (df_data['dir']+'/'+df_data['file']).tolist()
#     labels = tf.one_hot((df_data['class'].astype(int)).tolist(), num_classes)
#     return np.array([np.asarray(Image.open(fname)) for fname in files]), labels


# trash_data = train_data.loc[train_data.label=='trash']
# data_trash, trash_labels = prepare_jpg(trash_data)
# trash_ds = tf.data.Dataset.from_tensor_slices((data_trash, trash_labels)).batch(32)

# aug_dir = 'aug_images'
# if not os.path.isdir(aug_dir):
#     os.makedirs(aug_dir)

# for file in os.listdir(aug_dir):
#     os.remove(os.path.join(aug_dir, file))
    
# aug_factor = int(2*(train_data['class'].value_counts().max() - train_data['class'].value_counts().min()))
# for x, y in trash_ds:
#     gen = tf.keras.preprocessing.image.ImageDataGenerator(
#         zoom_range=0.3,
#         horizontal_flip=True,
#         rotation_range=90,
#         width_shift_range=.2,
#         height_shift_range=.2,
#         brightness_range=(0.25, 0.75)
#     )
#     gen.fit(x)
#     for x in gen.flow(x, batch_size=32, save_to_dir=aug_dir, save_prefix='aug', save_format='jpg'):
#         if len(os.listdir(aug_dir)) >= aug_factor:
#             print('generated %d augmented images'%(len(os.listdir(aug_dir))))
#             break
#     break


# trash_df = []
# for r, d, f in os.walk(aug_dir):
#     for file in f:
#         trash_df.append({'dir': r, 'file': file, 'path': os.path.join(r, file)})
        
# trash_df = pd.DataFrame(trash_df).sample(n=aug_factor)
# trash_df['label'] = 'trash'
# trash_df['class'] = str(classes.index('trash'))

# train_data = pd.concat([train_data, trash_df], axis=0)
# # append to train_ds
# train_ds = datagen_train.flow_from_dataframe(
#     train_data,  
#     x_col='path', 
#     y_col='class',
#     class_mode='categorical',
#     color_mode='rgb',
#     batch_size=32, 
#     shuffle=True,
#     save_format='jpg',
# )

# =======================
# model arquitecture
# =======================

# input
input_ = tf.keras.layers.Input(name='input', shape=(img_height, img_width, 3,))
# resize images
x = tf.keras.layers.experimental.preprocessing.Resizing(res_img_height, res_img_width)(input_)
# Rescale image values to [0, 1]
x = tf.keras.applications.xception.preprocess_input(x)

# base model
base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')
base_model.trainable = False
x = base_model(x, training=False)

# classification head
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu', name='dense_1')(x)
x = tf.keras.layers.Dense(512, activation='relu', name='dense_2')(x)

# output
output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs=[input_], outputs=[output])

for layer in base_model.layers:
    layer.trainable = False

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

# ===========================
# train
# =============================

hist, _ = np.histogram(train_data['class'].astype(np.int8), bins=np.arange(7)-0.5)
class_weights = hist.min()/hist
class_weights = {i:class_weights[i] for i in range(num_classes)}
print(class_weights)

model_dir = os.path.join('models', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
chkpt_dir = os.path.join(model_dir, 'checkpoint')

if not os.path.isdir(chkpt_dir): os.makedirs(chkpt_dir)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    # tf.keras.callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', patience=7, min_lr=1e-6, factor=0.1, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(chkpt_dir, monitor='val_categorical_accuracy', save_best_only=True)
]

epochs = 100
results = model.fit(train_ds, epochs=epochs, validation_data=(val_ds), callbacks=callbacks, class_weight=class_weights)

# unfreeze base model
model.get_layer('xception').trainable = True
# freeze classification head
# model.get_layer('dense_1').trainable = False
# model.get_layer('dense_2').trainable = False

epochs = 2
model.fit(train_ds, epochs=epochs, validation_data=(val_ds), callbacks=callbacks, class_weight=class_weights)

# reduce lr
model.optimizer.lr = 1e-4

epochs = 2
model.fit(train_ds, epochs=epochs, validation_data=(val_ds), callbacks=callbacks, class_weight=class_weights)


# =======================
# evaluation
# =======================

test_pred = model.predict(test_ds)
test_pred = np.argmax(test_pred, axis=1)
test_true = test_data['class'].astype(np.int8).to_numpy()

print(classification_report(test_true, test_pred))
conf_mat = confusion_matrix(test_true, test_pred, normalize='true')
conf_map = pd.DataFrame(columns=classes, data=conf_mat, index=classes)
conf_map
sns.heatmap(conf_mat, annot=True, xticklabels=classes, yticklabels=classes, cmap="YlGnBu", cbar=False);

# ==================
# save trained model
# ==================
save_dir = os.path.join(model_dir, 'best_model')
tf.keras.models.save_model(model, save_dir)