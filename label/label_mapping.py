import pickle

label_to_encoded_format = {
    'Du lịch': 0,
    'Nhà đất': 1,
    'Mua sắm': 2,
    'Tài chính': 3,
    'Mạng internet và viễn thông': 4,
    'Nhà và vườn': 5,
    'Kinh doanh và công nghiệp': 6,
    'Nghệ thuật': 7,
    'Giáo dục': 8,
    'Làm đẹp và thể hình': 9,
    'Con người và xã hội': 10,
    'Sách': 11,
    'Chính trị': 12,
    'Đồ ăn và đồ uống': 13,
    'Giao thông': 14,
    'Thói quen và sở thích': 15,
    'Giải trí': 16,
    'Sức khoẻ và bệnh tật': 17,
    'Pháp luật': 18,
    'Khoa học': 19,
    'Máy tính và thiết bị điện tử': 20,
    'Công nghệ mới': 21,
    'Thể thao': 22
}

# Tạo ánh xạ ngược
encoded_to_label = {v: k for k, v in label_to_encoded_format.items()}

# Gộp lại thành dict tổng
label_mappings = {
    "label2id": label_to_encoded_format,
    "id2label": encoded_to_label
}

# Ghi ra file
with open("label_mappings.pkl", "wb") as f:
    pickle.dump(label_mappings, f)

print(" Đã tạo file label_mappings.pkl thành công.")
