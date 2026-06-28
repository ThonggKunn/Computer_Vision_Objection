"""
Config dùng chung cho toàn bộ Vietnam Landmark Pipeline.
Import file này ở đầu mỗi step:
    from config import *
"""
from pathlib import Path

PROJECT_ROOT = Path("/Users/1pro/PycharmProjects/CV")
DATA_DIR     = Path("/Data/hanoi_landmark")

RAW_DIR      = DATA_DIR / "raw_images"
CLEAN_DIR    = DATA_DIR / "clean_images"
TRASH_DIR    = DATA_DIR / "trash"
DATASET_DIR  = DATA_DIR / "dataset"

TARGET_PER_CLASS = 1000
TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
TEST_RATIO        = 0.15
TARGET_SIZE      = (224, 224)
SEED             = 42

LANDMARKS = {
    "ho_guom":      {"id": 0, "name_vi": "H\u1ed3 G\u01b0\u01a1m",       "name_en": "Hoan Kiem Lake"},
    "chua_mot_cot": {"id": 1, "name_vi": "Ch\u00f9a M\u1ed9t C\u1ed9t",  "name_en": "One Pillar Pagoda"},
    "lang_bac":     {"id": 2, "name_vi": "L\u0103ng B\u00e1c",            "name_en": "Ho Chi Minh Mausoleum"},
    "ha_long":      {"id": 3, "name_vi": "V\u1ecbnh H\u1ea1 Long",        "name_en": "Ha Long Bay"},
    "hue":          {"id": 4, "name_vi": "C\u1ed1 \u0111\u00f4 Hu\u1ebf","name_en": "Hue Imperial City"},
    "cau_vang":     {"id": 5, "name_vi": "C\u1ea7u V\u00e0ng",            "name_en": "Golden Bridge"},
    "hoi_an":       {"id": 6, "name_vi": "Ph\u1ed1 c\u1ed5 H\u1ed9i An", "name_en": "Hoi An Ancient Town"},
    "duc_ba":       {"id": 7, "name_vi": "Nh\u00e0 th\u1edd \u0110\u1ee9c B\u00e0", "name_en": "Notre Dame Cathedral"},
    "ben_nha_rong": {"id": 8, "name_vi": "B\u1ebfn Nh\u00e0 R\u1ed3ng",  "name_en": "Nha Rong Wharf"},
    "ben_thanh":    {"id": 9, "name_vi": "Ch\u1ee3 B\u1ebfn Th\u00e0nh", "name_en": "Ben Thanh Market"},
}

# icrawler search queries mỗi class
QUERIES = {
    "ho_guom":      ["Ho Guom Hanoi", "Hoan Kiem Lake", "Ho Guom pagoda"],
    "chua_mot_cot": ["Chua Mot Cot Hanoi", "One Pillar Pagoda Vietnam"],
    "lang_bac":     ["Lang Bac Ho Chi Minh Mausoleum", "Ho Chi Minh tomb Hanoi"],
    "ha_long":      ["Ha Long Bay Vietnam", "Halong Bay karst", "Vinh Ha Long"],
    "hue":          ["Hue Imperial City", "Hue Citadel Vietnam", "Co do Hue"],
    "cau_vang":     ["Golden Bridge Da Nang", "Cau Vang Ba Na Hills"],
    "hoi_an":       ["Hoi An Ancient Town", "Hoi An lanterns", "Hoi An old town"],
    "duc_ba":       ["Notre Dame Cathedral Saigon", "Duc Ba church Ho Chi Minh"],
    "ben_nha_rong": ["Ben Nha Rong Saigon", "Dragon House Wharf Ho Chi Minh"],
    "ben_thanh":    ["Ben Thanh Market Saigon", "Cho Ben Thanh Ho Chi Minh"],
}
