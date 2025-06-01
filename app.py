# from flask import Flask, render_template, request, redirect, url_for
# from werkzeug.utils import secure_filename
# import os
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # Load mô hình
# model = load_model('model/skin_cancer_model.h5')
# disease_info = {
#     'actinic keratosis': {
#         'description': 'Tổn thương tiền ung thư do phơi nắng lâu dài, xuất hiện ở vùng da hở. Nên được kiểm tra định kỳ để phòng ngừa chuyển thành ung thư.',
#         'image': 'actinic_keratosis.jpg',
#         'article': 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis/',
#         'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
#     },
#     'basal cell carcinoma': {
#         'description': 'Loại ung thư da phổ biến nhất, phát triển chậm nhưng có thể phá huỷ mô nếu không điều trị. Ít khi di căn.',
#         'image': 'basal_cell_carcinoma.jpg',
#         'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
#         'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
#     },
#     'dermatofibroma': {
#         'description': 'U lành tính thường gặp ở chân hoặc tay. Không nguy hiểm nhưng có thể được loại bỏ nếu gây khó chịu.',
#         'image': 'dermatofibroma.jpg',
#         'article': 'https://www.dermnetnz.org/topics/dermatofibroma',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'melanoma': {
#         'description': 'Loại ung thư da nghiêm trọng nhất, có thể di căn nhanh. Cần phát hiện sớm và điều trị kịp thời.',
#         'image': 'melanoma_example.jpg',
#         'article': 'https://www.cancer.org/cancer/melanoma-skin-cancer.html',
#         'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
#     },
#     'nevus': {
#         'description': 'Nốt ruồi là tổn thương sắc tố lành tính. Nên theo dõi nếu có thay đổi màu sắc, kích thước hoặc chảy máu.',
#         'image': 'nevus_example.jpg',
#         'article': 'https://www.aad.org/public/diseases/a-z/moles',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'pigmented benign keratosis': {
#         'description': 'Tổn thương da có sắc tố lành tính, thường gặp ở người lớn tuổi. Không nguy hiểm.',
#         'image': 'pigmented_keratosis.jpg',
#         'article': 'https://www.dermnetnz.org/topics/seborrhoeic-keratosis',
#         'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
#     },
#     'seborrheic keratosis': {
#         'description': 'U da lành tính thường xuất hiện theo tuổi tác. Có thể loại bỏ bằng đốt laser nếu gây mất thẩm mỹ.',
#         'image': 'seborrheic_keratosis.jpg',
#         'article': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/',
#         'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
#     },
#     'squamous cell carcinoma': {
#         'description': 'Ung thư tế bào vảy, có nguy cơ lan rộng nếu không được điều trị. Phát hiện sớm giúp tăng hiệu quả điều trị.',
#         'image': 'squamous_cell_carcinoma.jpg',
#         'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'vascular lesion': {
#         'description': 'Tổn thương mạch máu như u máu, chấm đỏ. Thường lành tính nhưng cần theo dõi nếu to dần hoặc lan rộng.',
#         'image': 'vascular_lesion.jpg',
#         'article': 'https://www.dermnetnz.org/topics/vascular-tumours',
#         'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
#     }
# }





# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import json
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# # --- Load mô hình đã lưu ---
# model_path = 'model/skin_cancer_model.h5'  # Đường dẫn model bạn đã lưu
# model = tf.keras.models.load_model(model_path)

# # --- Load metadata (label_map, mean, std) ---
# metadata_path = 'model/metadata.json'
# with open(metadata_path, 'r', encoding='utf-8') as f:
#     metadata = json.load(f)

# label_map = {int(k): v for k, v in metadata['label_map'].items()}
# mean = np.array(metadata['mean'])
# std = np.array(metadata['std'])

# # --- Hàm tiền xử lý ảnh ---
# def preprocess_image(image_path, target_size=(128, 128)):
#     img = Image.open(image_path).convert('RGB').resize(target_size)
#     img_array = np.array(img).astype(np.float32)
#     # Chuẩn hóa ảnh: (img - mean) / std
#     img_array = (img_array - mean) / std
#     # Mô hình cần input shape (1, H, W, C)
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # --- Dự đoán ---
# def predict_image(model, image_path):
#     img_preprocessed = preprocess_image(image_path)
#     preds = model.predict(img_preprocessed)
#     class_idx = np.argmax(preds, axis=1)[0]
#     class_name = label_map[class_idx]
#     confidence = preds[0][class_idx]
#     return class_name, confidence

# # # Ví dụ sử dụng

# print(std,mean)
# print(label_map)

# # image_to_predict = 'model/ISIC_0000002.jpg'  # Đường dẫn ảnh muốn dự đoán
# def resize_image_array(image_path):
#     return np.asarray(Image.open(image_path).resize((128, 128)))

# # test_img = resize_image_array(image_to_predict)
    
# # # Display the test image
# # plt.figure(figsize=(6, 6))
# # plt.imshow(test_img)
# # plt.title("Test Image")
# # plt.axis('off')
# # plt.show()

# #     # Preprocess for prediction
# # test_img = np.expand_dims(test_img, axis=0)
# # test_img = (test_img - mean) / std
    
# # # Predict
# # prediction = model.predict(test_img)
# # predicted_class = np.argmax(prediction, axis=1)[0]
# # predicted_label = label_map[predicted_class]
# # # Show prediction
# # print(f"Predicted Class: {predicted_label}")
# # print(f"Confidence: {prediction[0][predicted_class]:.4f}")
    
# # # Show prediction distribution
# # plt.figure(figsize=(10, 6))
# # sns.barplot(x=list(label_map.values()), y=prediction[0])
# # plt.xlabel('Classes')
# # plt.ylabel('Probability')
# # plt.title('Prediction Probability Distribution')
# # plt.xticks(rotation=45, ha='right')
# # plt.tight_layout()
# # plt.show()

# def predict_image(img_path):
#     img  = resize_image_array(img_path)
#     img = (img - mean) / std
#     img = np.expand_dims(img, axis=0)
#     predictions = model.predict(img)
#     class_index = np.argmax(predictions)
#     confidence = np.max(predictions)
#     return label_map[class_index], confidence

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return redirect(request.url)
#         file = request.files['image']
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)
#             prediction, confidence = predict_image(filepath)
#             return render_template('index.html', image=filename, prediction=prediction, confidence=confidence)
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, redirect, url_for, session
# from werkzeug.utils import secure_filename
# import os
# import numpy as np
# from PIL import Image
# import json
# import tensorflow as tf

# app = Flask(__name__)
# app.secret_key = 'secret_key_for_session'  # dùng cho session
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # Load model và metadata
# model = tf.keras.models.load_model('model/skin_cancer_model.h5')

# with open('model/metadata.json', 'r', encoding='utf-8') as f:
#     metadata = json.load(f)

# label_map = {int(k): v for k, v in metadata['label_map'].items()}
# mean = np.array(metadata['mean'])
# std = np.array(metadata['std'])

# disease_info = {
#     'actinic keratosis': {
#         'name': 'Actinic Keratosis – Dày sừng ánh sáng (tổn thương tiền ung thư)',
#         'description': 'Actinic keratosis là một dạng tổn thương da tiền ung thư, hình thành do tiếp xúc lâu dài với tia cực tím từ ánh nắng mặt trời. Chúng thường xuất hiện dưới dạng các mảng da thô ráp, có vảy, màu đỏ hoặc nâu, tập trung ở các vùng da hở như mặt, tai, tay và cổ. Nếu không được theo dõi và điều trị kịp thời, khoảng 10% có thể tiến triển thành ung thư da tế bào vảy. Việc kiểm tra định kỳ giúp phát hiện sớm và điều trị hiệu quả.',
#         'image': 'actinic_keratosis.jpg',
#         'article': 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis/',
#         'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
#     },
#     'basal cell carcinoma': {
#         'name': 'Basal Cell Carcinoma – Ung thư biểu mô tế bào đáy',
#         'description': 'Ung thư biểu mô tế bào đáy là loại ung thư da phổ biến nhất nhưng ít có khả năng di căn. Nó phát triển chậm, thường xuất hiện dưới dạng một khối u nhỏ màu hồng nhạt, đôi khi bóng hoặc có vảy. Nếu không điều trị, nó có thể ăn sâu vào mô xung quanh gây biến dạng nghiêm trọng. Phát hiện sớm giúp loại bỏ hoàn toàn mà không để lại biến chứng.',
#         'image': 'basal_cell_carcinoma.jpg',
#         'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
#         'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
#     },
#     'dermatofibroma': {
#         'name': 'Dermatofibroma – U xơ da',
#         'description': 'Dermatofibroma là một loại u lành tính phổ biến của da, thường có màu nâu hoặc xám, nhỏ, chắc, không đau, xuất hiện chủ yếu ở chân hoặc tay. Tuy không gây nguy hiểm nhưng chúng có thể gây ngứa hoặc mất thẩm mỹ. Trong một số trường hợp, nếu u lớn hoặc gây khó chịu, có thể được loại bỏ bằng phẫu thuật.',
#         'image': 'dermatofibroma.jpg',
#         'article': 'https://www.dermnetnz.org/topics/dermatofibroma',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'melanoma': {
#         'name': 'Melanoma – U hắc tố ác tính',  
#         'description': 'Melanoma là loại ung thư da nghiêm trọng nhất, xuất phát từ các tế bào tạo sắc tố (melanocyte). Nó có khả năng di căn nhanh đến các cơ quan khác nếu không được phát hiện và điều trị sớm. Dấu hiệu cảnh báo bao gồm nốt ruồi thay đổi về hình dạng, kích thước, màu sắc, hoặc xuất hiện mới sau tuổi trưởng thành. Kiểm tra định kỳ giúp phát hiện kịp thời và cải thiện tỉ lệ sống sót.',
#         'image': 'melanoma_example.jpg',
#         'article': 'https://www.cancer.org/cancer/melanoma-skin-cancer.html',
#         'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
#     },
#     'nevus': {
#         'name': 'Nevus – Nốt ruồi',
#         'description': 'Nevus là tên y học của nốt ruồi – một tổn thương sắc tố lành tính. Chúng có thể xuất hiện ngay từ nhỏ hoặc hình thành theo thời gian. Phần lớn các nốt ruồi là lành tính, tuy nhiên, một số có thể biến đổi ác tính thành melanoma. Vì vậy, cần theo dõi nếu nốt ruồi có thay đổi bất thường như lớn nhanh, chảy máu, hoặc màu không đều.',
#         'image': 'nevus_example.jpg',
#         'article': 'https://www.aad.org/public/diseases/a-z/moles',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'pigmented benign keratosis': {
#         'name': 'Pigmented Benign Keratosis – Dày sừng lành tính có sắc tố',
#         'description': 'Là một dạng tổn thương sắc tố lành tính thường gặp ở người lớn tuổi, có thể nhầm lẫn với u ác tính do màu sắc đậm và hình dạng không đều. Thường không cần điều trị nếu không gây khó chịu hoặc ảnh hưởng thẩm mỹ. Bác sĩ da liễu có thể sử dụng dermoscopy hoặc sinh thiết để phân biệt với tổn thương ác tính nếu nghi ngờ.',
#         'image': 'pigmented_keratosis.jpg',
#         'article': 'https://www.dermnetnz.org/topics/seborrhoeic-keratosis',
#         'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
#     },
#     'seborrheic keratosis': {
#         'name': 'Seborrheic Keratosis – Dày sừng tiết bã',
#         'description': 'Seborrheic keratosis là khối u lành tính của da, thường xuất hiện theo tuổi tác, có màu nâu sẫm, đen hoặc vàng nhạt và bề mặt sần sùi. Mặc dù không nguy hiểm, chúng có thể bị kích ứng hoặc bị hiểu nhầm là ung thư da. Điều trị thường là thẩm mỹ, bao gồm đốt điện, áp lạnh hoặc laser nếu cần thiết.',
#         'image': 'seborrheic_keratosis.jpg',
#         'article': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/',
#         'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
#     },
#     'squamous cell carcinoma': {
#         'name': 'Squamous Cell Carcinoma – Ung thư biểu mô tế bào',
#         'description': 'Là loại ung thư da phổ biến thứ hai sau ung thư tế bào đáy, squamous cell carcinoma có thể phát triển ở bất kỳ vùng da nào, nhưng thường gặp ở vùng tiếp xúc ánh nắng nhiều. Nó có thể lan rộng nếu không điều trị, gây phá huỷ mô và di căn. Dấu hiệu bao gồm mảng da dày, có vảy, loét không lành. Phát hiện và điều trị sớm rất quan trọng.',
#         'image': 'squamous_cell_carcinoma.jpg',
#         'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'vascular lesion': {
#         'name': 'Vascular Lesion – Tổn thương mạch máu',
#         'description': 'Tổn thương mạch máu (vascular lesions) bao gồm u máu, vết chàm đỏ hoặc các đốm xuất huyết dưới da. Hầu hết đều lành tính và không cần điều trị. Tuy nhiên, nếu chúng phát triển nhanh, thay đổi màu sắc hoặc gây chảy máu, nên được đánh giá bởi bác sĩ chuyên khoa. Một số trường hợp có thể điều trị bằng laser thẩm mỹ.',
#         'image': 'vascular_lesion.jpg',
#         'article': 'https://www.dermnetnz.org/topics/vascular-tumours',
#         'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
#     }
# }


# # Hàm resize + chuẩn hoá ảnh
# def preprocess_image(image_path, target_size=(128, 128)):
#     img = Image.open(image_path).convert('RGB').resize(target_size)
#     img_array = np.array(img).astype(np.float32)
#     img_array = (img_array - mean) / std
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # Hàm dự đoán
# def predict_image(image_path):
#     img = preprocess_image(image_path)
#     predictions = model.predict(img)
#     class_idx = np.argmax(predictions)
#     confidence = predictions[0][class_idx]
#     class_name = label_map[class_idx]
#     return class_name, float(confidence)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if 'history' not in session:
#         session['history'] = []

#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return redirect(request.url)

#         file = request.files['image']
#         if file.filename == '':
#             return redirect(request.url)

#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             prediction, confidence = predict_image(filepath)

#             # Gợi ý thông tin
#             suggestion = disease_info.get(prediction, None)

#             # Lưu lịch sử
#             history = session['history']
#             history.insert(0, {'image': filename, 'prediction': prediction, 'confidence': confidence})
#             session['history'] = history[:10]  # Giới hạn 10 ảnh

#             return render_template(
#                 'index.html',
#                 image=filename,
#                 prediction=prediction,
#                 confidence=confidence,
#                 suggestion=suggestion,
#                 history=session['history'],
#             )

#     return render_template('index.html', history=session.get('history', []))

# @app.route('/clear_history')
# def clear_history():
#     session.pop('history', None)
#     return redirect(url_for('index'))

# if __name__ == '__main__':
#     if not os.path.exists(app.config['UPLOAD_FOLDER']):
#         os.makedirs(app.config['UPLOAD_FOLDER'])
#     app.run(debug=True)

# from flask import Flask, render_template, request, redirect, url_for, session
# from werkzeug.utils import secure_filename
# import os
# import numpy as np
# from PIL import Image
# import json
# import tensorflow as tf
# from flask import Flask, request, jsonify
# from openai import OpenAI

# app = Flask(__name__)
# app.secret_key = 'secret_key_for_session'  # dùng cho session
# app.config['UPLOAD_FOLDER'] = 'static/uploads'

# # Load model và metadata
# # model = tf.keras.models.load_model('model/skin_cancer_model.h5')
# model = tf.keras.models.load_model('model/skin_cancer_model_CNN_Custom.h5')


# with open('model/metadata.json', 'r', encoding='utf-8') as f:
#     metadata = json.load(f)

# label_map = {int(k): v for k, v in metadata['label_map'].items()}
# mean = np.array(metadata['mean'])
# std = np.array(metadata['std'])

# disease_info = {
#     'actinic keratosis': {
#         'name': 'Actinic Keratosis – Dày sừng ánh sáng (tổn thương tiền ung thư)',
#         'description': 'Actinic keratosis là một dạng tổn thương da tiền ung thư, hình thành do tiếp xúc lâu dài với tia cực tím từ ánh nắng mặt trời. Chúng thường xuất hiện dưới dạng các mảng da thô ráp, có vảy, màu đỏ hoặc nâu, tập trung ở các vùng da hở như mặt, tai, tay và cổ. Nếu không được theo dõi và điều trị kịp thời, khoảng 10% có thể tiến triển thành ung thư da tế bào vảy. Việc kiểm tra định kỳ giúp phát hiện sớm và điều trị hiệu quả.',
#         'image': 'actinic_keratosis.jpg',
#         'article': 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis/',
#         'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
#     },
#     'basal cell carcinoma': {
#         'name': 'Basal Cell Carcinoma – Ung thư biểu mô tế bào đáy',
#         'description': 'Ung thư biểu mô tế bào đáy là loại ung thư da phổ biến nhất nhưng ít có khả năng di căn. Nó phát triển chậm, thường xuất hiện dưới dạng một khối u nhỏ màu hồng nhạt, đôi khi bóng hoặc có vảy. Nếu không điều trị, nó có thể ăn sâu vào mô xung quanh gây biến dạng nghiêm trọng. Phát hiện sớm giúp loại bỏ hoàn toàn mà không để lại biến chứng.',
#         'image': 'basal_cell_carcinoma.jpg',
#         'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
#         'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
#     },
#     'dermatofibroma': {
#         'name': 'Dermatofibroma – U xơ da',
#         'description': 'Dermatofibroma là một loại u lành tính phổ biến của da, thường có màu nâu hoặc xám, nhỏ, chắc, không đau, xuất hiện chủ yếu ở chân hoặc tay. Tuy không gây nguy hiểm nhưng chúng có thể gây ngứa hoặc mất thẩm mỹ. Trong một số trường hợp, nếu u lớn hoặc gây khó chịu, có thể được loại bỏ bằng phẫu thuật.',
#         'image': 'dermatofibroma.jpg',
#         'article': 'https://www.dermnetnz.org/topics/dermatofibroma',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'melanoma': {
#         'name': 'Melanoma – U hắc tố ác tính',  
#         'description': 'Melanoma là loại ung thư da nghiêm trọng nhất, xuất phát từ các tế bào tạo sắc tố (melanocyte). Nó có khả năng di căn nhanh đến các cơ quan khác nếu không được phát hiện và điều trị sớm. Dấu hiệu cảnh báo bao gồm nốt ruồi thay đổi về hình dạng, kích thước, màu sắc, hoặc xuất hiện mới sau tuổi trưởng thành. Kiểm tra định kỳ giúp phát hiện kịp thời và cải thiện tỉ lệ sống sót.',
#         'image': 'melanoma_example.jpg',
#         'article': 'https://www.cancer.org/cancer/melanoma-skin-cancer.html',
#         'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
#     },
#     'nevus': {
#         'name': 'Nevus – Nốt ruồi',
#         'description': 'Nevus là tên y học của nốt ruồi – một tổn thương sắc tố lành tính. Chúng có thể xuất hiện ngay từ nhỏ hoặc hình thành theo thời gian. Phần lớn các nốt ruồi là lành tính, tuy nhiên, một số có thể biến đổi ác tính thành melanoma. Vì vậy, cần theo dõi nếu nốt ruồi có thay đổi bất thường như lớn nhanh, chảy máu, hoặc màu không đều.',
#         'image': 'nevus_example.jpg',
#         'article': 'https://www.aad.org/public/diseases/a-z/moles',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'pigmented benign keratosis': {
#         'name': 'Pigmented Benign Keratosis – Dày sừng lành tính có sắc tố',
#         'description': 'Là một dạng tổn thương sắc tố lành tính thường gặp ở người lớn tuổi, có thể nhầm lẫn với u ác tính do màu sắc đậm và hình dạng không đều. Thường không cần điều trị nếu không gây khó chịu hoặc ảnh hưởng thẩm mỹ. Bác sĩ da liễu có thể sử dụng dermoscopy hoặc sinh thiết để phân biệt với tổn thương ác tính nếu nghi ngờ.',
#         'image': 'pigmented_keratosis.jpg',
#         'article': 'https://www.dermnetnz.org/topics/seborrhoeic-keratosis',
#         'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
#     },
#     'seborrheic keratosis': {
#         'name': 'Seborrheic Keratosis – Dày sừng tiết bã',
#         'description': 'Seborrheic keratosis là khối u lành tính của da, thường xuất hiện theo tuổi tác, có màu nâu sẫm, đen hoặc vàng nhạt và bề mặt sần sùi. Mặc dù không nguy hiểm, chúng có thể bị kích ứng hoặc bị hiểu nhầm là ung thư da. Điều trị thường là thẩm mỹ, bao gồm đốt điện, áp lạnh hoặc laser nếu cần thiết.',
#         'image': 'seborrheic_keratosis.jpg',
#         'article': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/',
#         'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
#     },
#     'squamous cell carcinoma': {
#         'name': 'Squamous Cell Carcinoma – Ung thư biểu mô tế bào',
#         'description': 'Là loại ung thư da phổ biến thứ hai sau ung thư tế bào đáy, squamous cell carcinoma có thể phát triển ở bất kỳ vùng da nào, nhưng thường gặp ở vùng tiếp xúc ánh nắng nhiều. Nó có thể lan rộng nếu không điều trị, gây phá huỷ mô và di căn. Dấu hiệu bao gồm mảng da dày, có vảy, loét không lành. Phát hiện và điều trị sớm rất quan trọng.',
#         'image': 'squamous_cell_carcinoma.jpg',
#         'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
#         'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
#     },
#     'vascular lesion': {
#         'name': 'Vascular Lesion – Tổn thương mạch máu',
#         'description': 'Tổn thương mạch máu (vascular lesions) bao gồm u máu, vết chàm đỏ hoặc các đốm xuất huyết dưới da. Hầu hết đều lành tính và không cần điều trị. Tuy nhiên, nếu chúng phát triển nhanh, thay đổi màu sắc hoặc gây chảy máu, nên được đánh giá bởi bác sĩ chuyên khoa. Một số trường hợp có thể điều trị bằng laser thẩm mỹ.',
#         'image': 'vascular_lesion.jpg',
#         'article': 'https://www.dermnetnz.org/topics/vascular-tumours',
#         'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
#     }
# }

# def preprocess_image(image_path, target_size=(128, 128)):
#     img = Image.open(image_path).convert('RGB').resize(target_size)
#     img_array = np.array(img).astype(np.float32)
#     img_array = (img_array - mean) / std
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# def predict_image(image_path):
#     img_preprocessed = preprocess_image(image_path)
#     preds = model.predict(img_preprocessed)
#     class_idx = np.argmax(preds, axis=1)[0]
#     class_name = label_map[class_idx]
#     confidence = preds[0][class_idx]
#     return class_name, float(confidence)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         if 'image' not in request.files:
#             return redirect(request.url)
#         file = request.files['image']
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(filepath)

#             prediction, confidence = predict_image(filepath)

#             # Lưu lịch sử dự đoán vào session
#             history = session.get('history', [])
#             history.append({
#                 'image': filename,
#                 'prediction': prediction,
#                 'confidence': confidence
#             })
#             # Giới hạn lưu tối đa 10 mục gần nhất
#             history = history[-10:]
#             session['history'] = history

#             # Lấy thông tin mô tả bệnh
#             suggestion = disease_info.get(prediction)

#             return render_template('index.html', image=filename, prediction=prediction, confidence=confidence, suggestion=suggestion)
#     return render_template('index.html')

# @app.route('/history')
# def history():
#     history = session.get('history', [])
#     return render_template('history.html', history=history)

# @app.route('/clear_history')
# def clear_history():
#     session.pop('history', None)
#     return redirect(url_for('history'))

# if __name__ == '__main__':
#     app.run(debug=True)

# app = Flask(__name__)

# client = OpenAI(api_key="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")

# @app.route('/chatbot', methods=['POST'])
# def chatbot():
#     data = request.json
#     question = data.get('question')
#     if not question:
#         return jsonify({'answer': 'Vui lòng nhập câu hỏi.'})

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "user", "content": question}
#             ]
#         )
#         answer = response.choices[0].message.content
#         return jsonify({'answer': answer})
#     except Exception as e:
#         print("Error:", e)
#         return jsonify({'answer': 'Có lỗi khi gọi API, vui lòng thử lại sau.'})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from openai import OpenAI

app = Flask(__name__)
app.secret_key = 'secret_key_for_session'  # Session key
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load model và metadata
model = tf.keras.models.load_model('model/skin_cancer_model_CNN_Custom.h5')

with open('model/metadata.json', 'r', encoding='utf-8') as f:
    metadata = json.load(f)

label_map = {int(k): v for k, v in metadata['label_map'].items()}
mean = np.array(metadata['mean'])
std = np.array(metadata['std'])

disease_info = {
    'actinic keratosis': {
        'name': 'Actinic Keratosis – Dày sừng ánh sáng (tổn thương tiền ung thư)',
        'description': 'Actinic keratosis là một dạng tổn thương da tiền ung thư, hình thành do tiếp xúc lâu dài với tia cực tím từ ánh nắng mặt trời. Chúng thường xuất hiện dưới dạng các mảng da thô ráp, có vảy, màu đỏ hoặc nâu, tập trung ở các vùng da hở như mặt, tai, tay và cổ. Nếu không được theo dõi và điều trị kịp thời, khoảng 10% có thể tiến triển thành ung thư da tế bào vảy. Việc kiểm tra định kỳ giúp phát hiện sớm và điều trị hiệu quả.',
        'image': 'actinic_keratosis.jpg',
        'article': 'https://www.skincancer.org/skin-cancer-information/actinic-keratosis/',
        'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
    },
    'basal cell carcinoma': {
        'name': 'Basal Cell Carcinoma – Ung thư biểu mô tế bào đáy',
        'description': 'Ung thư biểu mô tế bào đáy là loại ung thư da phổ biến nhất nhưng ít có khả năng di căn. Nó phát triển chậm, thường xuất hiện dưới dạng một khối u nhỏ màu hồng nhạt, đôi khi bóng hoặc có vảy. Nếu không điều trị, nó có thể ăn sâu vào mô xung quanh gây biến dạng nghiêm trọng. Phát hiện sớm giúp loại bỏ hoàn toàn mà không để lại biến chứng.',
        'image': 'basal_cell_carcinoma.jpg',
        'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
        'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
    },
    'dermatofibroma': {
        'name': 'Dermatofibroma – U xơ da',
        'description': 'Dermatofibroma là một loại u lành tính phổ biến của da, thường có màu nâu hoặc xám, nhỏ, chắc, không đau, xuất hiện chủ yếu ở chân hoặc tay. Tuy không gây nguy hiểm nhưng chúng có thể gây ngứa hoặc mất thẩm mỹ. Trong một số trường hợp, nếu u lớn hoặc gây khó chịu, có thể được loại bỏ bằng phẫu thuật.',
        'image': 'dermatofibroma.jpg',
        'article': 'https://www.dermnetnz.org/topics/dermatofibroma',
        'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
    },
    'melanoma': {
        'name': 'Melanoma – U hắc tố ác tính',  
        'description': 'Melanoma là loại ung thư da nghiêm trọng nhất, xuất phát từ các tế bào tạo sắc tố (melanocyte). Nó có khả năng di căn nhanh đến các cơ quan khác nếu không được phát hiện và điều trị sớm. Dấu hiệu cảnh báo bao gồm nốt ruồi thay đổi về hình dạng, kích thước, màu sắc, hoặc xuất hiện mới sau tuổi trưởng thành. Kiểm tra định kỳ giúp phát hiện kịp thời và cải thiện tỉ lệ sống sót.',
        'image': 'melanoma_example.jpg',
        'article': 'https://www.cancer.org/cancer/melanoma-skin-cancer.html',
        'advice': 'https://booking.benhvien108.vn/dich-vu-kham/da-lieu'
    },
    'nevus': {
        'name': 'Nevus – Nốt ruồi',
        'description': 'Nevus là tên y học của nốt ruồi – một tổn thương sắc tố lành tính. Chúng có thể xuất hiện ngay từ nhỏ hoặc hình thành theo thời gian. Phần lớn các nốt ruồi là lành tính, tuy nhiên, một số có thể biến đổi ác tính thành melanoma. Vì vậy, cần theo dõi nếu nốt ruồi có thay đổi bất thường như lớn nhanh, chảy máu, hoặc màu không đều.',
        'image': 'nevus_example.jpg',
        'article': 'https://www.aad.org/public/diseases/a-z/moles',
        'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
    },
    'pigmented benign keratosis': {
        'name': 'Pigmented Benign Keratosis – Dày sừng lành tính có sắc tố',
        'description': 'Là một dạng tổn thương sắc tố lành tính thường gặp ở người lớn tuổi, có thể nhầm lẫn với u ác tính do màu sắc đậm và hình dạng không đều. Thường không cần điều trị nếu không gây khó chịu hoặc ảnh hưởng thẩm mỹ. Bác sĩ da liễu có thể sử dụng dermoscopy hoặc sinh thiết để phân biệt với tổn thương ác tính nếu nghi ngờ.',
        'image': 'pigmented_keratosis.jpg',
        'article': 'https://www.dermnetnz.org/topics/seborrhoeic-keratosis',
        'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
    },
    'seborrheic keratosis': {
        'name': 'Seborrheic Keratosis – Dày sừng tiết bã',
        'description': 'Seborrheic keratosis là khối u lành tính của da, thường xuất hiện theo tuổi tác, có màu nâu sẫm, đen hoặc vàng nhạt và bề mặt sần sùi. Mặc dù không nguy hiểm, chúng có thể bị kích ứng hoặc bị hiểu nhầm là ung thư da. Điều trị thường là thẩm mỹ, bao gồm đốt điện, áp lạnh hoặc laser nếu cần thiết.',
        'image': 'seborrheic_keratosis.jpg',
        'article': 'https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/',
        'advice': 'https://vinmec.com/vi/dich-vu/da-lieu-tham-my/'
    },
    'squamous cell carcinoma': {
        'name': 'Squamous Cell Carcinoma – Ung thư biểu mô tế bào',
        'description': 'Là loại ung thư da phổ biến thứ hai sau ung thư tế bào đáy, squamous cell carcinoma có thể phát triển ở bất kỳ vùng da nào, nhưng thường gặp ở vùng tiếp xúc ánh nắng nhiều. Nó có thể lan rộng nếu không điều trị, gây phá huỷ mô và di căn. Dấu hiệu bao gồm mảng da dày, có vảy, loét không lành. Phát hiện và điều trị sớm rất quan trọng.',
        'image': 'squamous_cell_carcinoma.jpg',
        'article': 'https://www.cancer.org/cancer/basal-and-squamous-cell-skin-cancer.html',
        'advice': 'https://tdc.medlatec.vn/chuyen-khoa/da-lieu'
    },
    'vascular lesion': {
        'name': 'Vascular Lesion – Tổn thương mạch máu',
        'description': 'Tổn thương mạch máu (vascular lesions) bao gồm u máu, vết chàm đỏ hoặc các đốm xuất huyết dưới da. Hầu hết đều lành tính và không cần điều trị. Tuy nhiên, nếu chúng phát triển nhanh, thay đổi màu sắc hoặc gây chảy máu, nên được đánh giá bởi bác sĩ chuyên khoa. Một số trường hợp có thể điều trị bằng laser thẩm mỹ.',
        'image': 'vascular_lesion.jpg',
        'article': 'https://www.dermnetnz.org/topics/vascular-tumours',
        'advice': 'https://benhvien108.vn/kham-chuyen-khoa/da-lieu'
    }
}

# OpenAI client
# client = OpenAI(api_key="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")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB').resize(target_size)
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array - mean) / std
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    img_preprocessed = preprocess_image(image_path)
    preds = model.predict(img_preprocessed)
    class_idx = np.argmax(preds, axis=1)[0]
    class_name = label_map[class_idx]
    confidence = preds[0][class_idx]
    return class_name, float(confidence)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            prediction, confidence = predict_image(filepath)

            # Lưu lịch sử dự đoán vào session
            history = session.get('history', [])
            history.append({
                'image': filename,
                'prediction': prediction,
                'confidence': confidence
            })
            history = history[-10:]
            session['history'] = history

            suggestion = disease_info.get(prediction)

            return render_template('index.html', image=filename, prediction=prediction, confidence=confidence, suggestion=suggestion)
    return render_template('index.html')

@app.route('/history')
def history():
    history = session.get('history', [])
    return render_template('history.html', history=history)

@app.route('/clear_history')
def clear_history():
    session.pop('history', None)
    return redirect(url_for('history'))

@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'answer': 'Vui lòng nhập câu hỏi.'})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content
        return jsonify({'answer': answer})
    except Exception as e:
        print("Error:", e)
        return jsonify({'answer': 'Có lỗi khi gọi API, vui lòng thử lại sau.'})

# if __name__ == '__main__':
#     app.run(debug=True)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
