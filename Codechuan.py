import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tensorflow.keras.applications import DenseNet201,DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import concurrent.futures
import multiprocessing
from tensorflow.keras.preprocessing.image import array_to_img
import gc
import os
import json


# Mount Drive trong Colab (chạy 1 lần)
from google.colab import drive
drive.mount('/content/drive')

# Thư mục lưu kết quả
save_dir = '/content/drive/MyDrive/SkinCancerResults'
os.makedirs(save_dir, exist_ok=True)


print("Số GPU khả dụng:", len(tf.config.list_physical_devices('GPU')))
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
print("Số lõi CPU:", multiprocessing.cpu_count())


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Acc Huấn luyện')
    ax1.plot(history.history['val_accuracy'], label='Acc Kiểm tra')
    ax1.set_title('Độ chính xác'); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history['loss'], label='Loss Huấn luyện')
    ax2.plot(history.history['val_loss'], label='Loss Kiểm tra')
    ax2.set_title('Hàm mất mát'); ax2.legend(); ax2.grid(True)
    plt.savefig(os.path.join(save_dir, 'training_history.png'))

    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán'); plt.ylabel('Thực tế'); plt.title('Ma trận nhầm lẫn')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))

    plt.show()

def plot_data_distribution(class_counts, label_map):
    plt.figure(figsize=(12,6))
    plt.bar(list(label_map.values()), class_counts)
    plt.xticks(rotation=45, ha='right')
    plt.title('Phân bố số lượng ảnh theo lớp')
    plt.ylabel('Số ảnh')
    plt.savefig(os.path.join(save_dir, 'data_distribution.png'))
    plt.show()

def plot_sample_images(df, label_map, num_samples=9):
    plt.figure(figsize=(12,12))
    for i, cls in enumerate(df['label'].unique()[:num_samples]):
        sample = df[df['label']==cls].sample(1).iloc[0]
        plt.subplot(3,3,i+1)
        plt.imshow(sample['image'])
        plt.title(label_map[cls]); plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_images.png'))

    plt.show()

def plot_roc_curves(y_test, y_pred_proba, num_classes, label_map):
    plt.figure(figsize=(12,10))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred_proba[:,i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f"{label_map[i]} (AUC={roc_auc[i]:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('Tỉ lệ dương tính giả'); plt.ylabel('Tỉ lệ dương tính đúng')
    plt.title('Đường cong ROC'); plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))

    plt.show()
    return np.mean(list(roc_auc.values()))


def create_dataframe(data_dir):
    data = [
        {"image_path": os.path.join(data_dir, cls_name, fname), "label": idx}
        for idx, cls_name in enumerate(os.listdir(data_dir))
        for fname in os.listdir(os.path.join(data_dir, cls_name))
    ]
    return pd.DataFrame(data)


# Hàm resize ảnh
def resize_image_array(image_path):
    return np.asarray(Image.open(image_path).resize((128, 128)))


def create_and_train_model(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs=25):
    # Tạo Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Xây dựng mô hình
    inp = Input(shape=input_shape)
    base = DenseNet201(include_top=False, weights='imagenet', input_tensor=inp)
    x = Flatten()(base.output)
    x = Dropout(0.7)(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    # x = Dense(512, activation='relu')(x)

    x = Dropout(0.5)(x)
    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    opt = SGD(learning_rate=1e-3, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    # Tính class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
    class_weight_dict = dict(enumerate(class_weights))

    # Callbacks
    lr_red = ReduceLROnPlateau(monitor='val_accuracy', patience=3, factor=0.5, min_lr=1e-5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    # Huấn luyện mô hình
    history = model.fit(
        # X_train, y_train,
        # batch_size=64,
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[lr_red, early_stopping],
        class_weight=class_weight_dict
    )
    return model, history

train_dir = '/content/input/Skin cancer ISIC The International Skin Imaging Collaboration/Train'
test_dir  = '/content/input/Skin cancer ISIC The International Skin Imaging Collaboration/Test'
max_per_class = 1500

df_train = create_dataframe(train_dir)
df_test = create_dataframe(test_dir)
df = pd.concat([df_train, df_test], ignore_index=True)

label_map = {i: cls for i, cls in enumerate(os.listdir(train_dir))}
num_classes = len(label_map)
print("Label map:", label_map)

df = df.groupby('label').head(max_per_class).reset_index(drop=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    df['image'] = list(executor.map(resize_image_array, df['image_path']))
df = df.dropna(subset=['image']).reset_index(drop=True)
X = np.stack(df['image'].values)
y = to_categorical(df['label'].values, num_classes=num_classes)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.5,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    fill_mode='nearest'
)

augmented_df = pd.DataFrame(columns=['image', 'label'])

for class_label in df['label'].unique():
    class_images = df[df['label'] == class_label]
    image_arrays = class_images['image'].values
    num_images_needed = max_per_class - len(image_arrays)

    augmented_df = pd.concat([augmented_df, class_images], ignore_index=True)

    if num_images_needed > 0:
        selected_images = np.random.choice(image_arrays, size=num_images_needed, replace=True)
        for img_array in selected_images:
            image_tensor = np.expand_dims(img_array, axis=0)
            augmented_img = next(datagen.flow(image_tensor, batch_size=1))[0].astype('uint8')
            new_row = pd.DataFrame([{'image': augmented_img, 'label': class_label}])
            augmented_df = pd.concat([augmented_df, new_row], ignore_index=True)

# Cắt bớt nếu quá giới hạn
# Cân bằng và shuffle
df = augmented_df.groupby('label').head(max_per_class).sample(frac=1, random_state=42).reset_index(drop=True)

counts = df['label'].value_counts().sort_index()
for i,name in label_map.items():
    print(f"{name:<30}: {counts[i]}")
plot_data_distribution(counts, label_map)
plot_sample_images(df, label_map)

X = np.stack(df['image'].values)  # (N,128,128,3)
y = df['label'].values

label_map_strkey = {str(k): v for k, v in label_map.items()}

# Tạo dict chung để lưu
data_to_save = {
    'label_map': label_map_strkey,
    'mean': mean.tolist(),
    'std': std.tolist()
}

# Đường dẫn file lưu
save_path = '/content/drive/MyDrive/SkinCancerResults/metadata.json'

# Lưu file JSON
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, indent=4, ensure_ascii=False)

print(f"Đã lưu label_map, mean, std vào file {save_path}")

# Chuẩn hóa
mean, std = X.mean(), X.std()
# X = (X - mean) / std
# X = X.astype('float32')

# # One-hot
# y_cat = tf.keras.utils.to_categorical(y, num_classes)

# # Chia tập
# X_train_val, X_test, y_train_val, y_test = train_test_split(
#     X, y_cat, test_size=0.2, random_state=42
# )
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_val, y_train_val, test_size=0.2, random_state=42
# )

# print("Shapes:", X_train.shape, X_val.shape, X_test.shape)

# df = df.drop(columns=['image'])
# X = None
# y = None
# gc.collect()

# input_shape = (128, 128, 3)
# model, history = create_and_train_model(
#     X_train, y_train, X_val, y_val,
#     input_shape, num_classes, epochs=25
# )
# plot_training_history(history)

# # Đánh giá mô hình
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_test_classes = np.argmax(y_test, axis=1)

# # Vẽ ma trận nhầm lẫn
# plot_confusion_matrix(y_test_classes, y_pred_classes, list(label_map.values()))

# # Vẽ đường cong ROC
# mean_auc = plot_roc_curves(y_test, y_pred, num_classes, label_map)
# print(f"Mean AUC: {mean_auc:.2f}")

# # Báo cáo phân loại
# print(classification_report(y_test_classes, y_pred_classes, target_names=list(label_map.values())))

# y_pred_proba = model.predict(X_test)

# from sklearn.metrics import precision_recall_curve

# plt.figure(figsize=(12,10))
# for i in range(num_classes):
#     prec, rec, _ = precision_recall_curve(y_test[:, i], y_pred_proba[:, i])
#     plt.plot(rec, prec, lw=2,
#              label=f"{label_map[i]} (AP={auc(rec, prec):.2f})")

# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision–Recall Curves')
# plt.legend(loc='lower left')
# plt.grid(True)
# plt.show()

# def predict_random_samples(model, X_test, y_test, label_map, num_samples=6):
#     # Chọn ngẫu nhiên các chỉ số mẫu
#     indices = np.random.choice(X_test.shape[0], num_samples, replace=False)
#     X_samples = X_test[indices]
#     y_samples = y_test[indices]
    
#     # Dự đoán
#     y_pred = model.predict(X_samples)
#     y_pred_classes = np.argmax(y_pred, axis=1)
#     y_true_classes = np.argmax(y_samples, axis=1)
    
#     # Tạo lưới hiển thị
#     rows = (num_samples + 2) // 3  # Số hàng (3 cột)
#     plt.figure(figsize=(12, 4 * rows))
    
#     for i in range(num_samples):
#         plt.subplot(rows, 3, i + 1)
#         img = array_to_img(X_samples[i])
#         plt.imshow(img)
#         plt.axis('off')
        
#         # Lấy nhãn và xác suất
#         true_label = label_map[y_true_classes[i]]
#         pred_label = label_map[y_pred_classes[i]]
#         pred_prob = y_pred[i][y_pred_classes[i]] * 100  # Xác suất lớp dự đoán
        
#         # Hiển thị tiêu đề
#         title = f"Thực tế: {true_label}\nDự đoán: {pred_label}\nXác suất: {pred_prob:.1f}%"
#         plt.title(title, fontsize=10, pad=10)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'random_sample_predictions.png'))
#     plt.show()

# # Chạy hàm với mô hình và dữ liệu
# predict_random_samples(model, X_test, y_test, label_map, num_samples=6)

# test_image_path = '/content/input/Skin cancer ISIC The International Skin Imaging Collaboration/Test/nevus/ISIC_0000015.jpg'
# test_img = resize_image_array(test_image_path)
    
# # Display the test image
# plt.figure(figsize=(6, 6))
# plt.imshow(test_img)
# plt.title("Test Image")
# plt.axis('off')
# plt.show()
    
#     # Preprocess for prediction
# test_img = np.expand_dims(test_img, axis=0)
# test_img = (test_img - mean) / std
    
# # Predict
# prediction = model.predict(test_img)
# predicted_class = np.argmax(prediction, axis=1)[0]
# predicted_label = label_map[predicted_class]
# # Show prediction
# print(f"Predicted Class: {predicted_label}")
# print(f"Confidence: {prediction[0][predicted_class]:.4f}")
    
# # Show prediction distribution
# plt.figure(figsize=(10, 6))
# sns.barplot(x=list(label_map.values()), y=prediction[0])
# plt.xlabel('Classes')
# plt.ylabel('Probability')
# plt.title('Prediction Probability Distribution')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()


# model.save(os.path.join(save_dir, 'skin_cancer_model.h5'))
# print(f"Đã lưu mô hình tại {os.path.join(save_dir, 'skin_cancer_model.h5')}")
