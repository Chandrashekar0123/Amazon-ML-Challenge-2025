# Smart Product Pricing - Colab-ready pipeline (TF-IDF + VGG16 + LightGBM + Ensemble)
# Cell markers (use in Colab as separate cells) are indicated by comments '# %%'

# %%
"""
Instructions:
1) Upload 'dataset/train.csv' and 'dataset/test.csv' to Colab (or mount Google Drive)
2) Run cells sequentially. Set SAMPLE_TRAIN to a small value to test (e.g., 5000)
3) After first run, cached images/features accelerate future runs.
"""

# %%
# Install required packages (run once in Colab)
!pip install -q torch torchvision tqdm lightgbm scikit-learn requests

# %%
# Imports
import os
import re
import time
import math
import random
import urllib
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

# ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import lightgbm as lgb

# Torch for VGG
import torch
from torchvision import models, transforms
from PIL import Image

# %%
# ----------------------------
# User parameters (tweak here)
# ----------------------------
TRAIN_PATH = '/content/drive/MyDrive/Amazon_ML/student_resource/dataset/train.csv'  # change if needed
TEST_PATH = '/content/drive/MyDrive/Amazon_ML/student_resource/dataset/test.csv'
DOWNLOAD_FOLDER = '/content/images_cache'
FEATURE_CACHE = '/content/features_cache'
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
os.makedirs(FEATURE_CACHE, exist_ok=True)

# For quick testing set small SAMPLE_TRAIN, then increase to 40k/75k for final run
SAMPLE_TRAIN = 5000   # set to None to use full train.csv
TFIDF_MAX_FEAT = 15000
PCA_IMG_DIM = 256
N_FOLDS = 5
SEED = 42
random.seed(SEED); np.random.seed(SEED)

# %%
# ----------------------------
# Utility functions
# ----------------------------

def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0.01, None)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom[denom == 0] = 1.0
    return np.mean(np.abs(y_pred - y_true) / denom) * 100.0


def extract_ipq(text):
    if not isinstance(text, str): return 1.0
    m = re.search(r'(?:IPQ:|Pack of|pack of)\s*(\d+)', text, re.I)
    if m: return float(m.group(1))
    m2 = re.search(r'(\d+)\s*(?:pcs|pieces|count|ct|pk|pack)', text, re.I)
    if m2: return float(m2.group(1))
    return 1.0


def safe_brand(text):
    if not isinstance(text, str): return 'Unknown'
    m = re.search(r'(?:Brand:|brand:)\s*([^\n,;|]+)', text)
    if m:
        b = re.sub(r'[^\w\s]', '', m.group(1)).strip().split()[0]
        return b if b else 'Unknown'
    tokens = text.strip().split()
    return tokens[0] if tokens else 'Unknown'

# Download helper with retries
import requests

def download_image(url, savefolder=DOWNLOAD_FOLDER, timeout=10, max_retries=3):
    if not isinstance(url, str) or url.strip() == '':
        return None
    filename = Path(url).name.split('?')[0]
    savepath = os.path.join(savefolder, filename)
    if os.path.exists(savepath):
        return savepath
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                with open(savepath, 'wb') as f:
                    f.write(r.content)
                return savepath
        except Exception as e:
            time.sleep(0.5 + attempt)
    return None

# Multiprocessing wrapper for downloads
def parallel_download(urls, savefolder=DOWNLOAD_FOLDER, workers=12):
    os.makedirs(savefolder, exist_ok=True)
    with Pool(workers) as p:
        results = list(tqdm(p.imap(partial(download_image, savefolder=savefolder), urls), total=len(urls)))
    return results

# %%
# ----------------------------
# Load data
# ----------------------------
print('Loading data...')
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
print('Train rows:', len(train), 'Test rows:', len(test))
if SAMPLE_TRAIN is not None and SAMPLE_TRAIN < len(train):
    train = train.sample(SAMPLE_TRAIN, random_state=SEED).reset_index(drop=True)
    print('Using sample of train:', len(train))

# Basic cleaning
train['catalog_content'] = train['catalog_content'].fillna('').astype(str)
test['catalog_content'] = test['catalog_content'].fillna('').astype(str)

# ----------------------------
# Text features: TF-IDF + Ridge
# ----------------------------
print('Building TF-IDF...')
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=(1,2), stop_words='english', min_df=3)
X_text_train = tfidf.fit_transform(train['catalog_content'])
X_text_test = tfidf.transform(test['catalog_content'])

# log-target
y_log = np.log1p(train['price'].astype(float).values)

print('Training Ridge on TF-IDF (text only)...')
ridge = Ridge(alpha=1.0, random_state=SEED)
ridge.fit(X_text_train, y_log)
ridge_oof = ridge.predict(X_text_train)
ridge_preds_test = ridge.predict(X_text_test)
print('Ridge OOF SMAPE:', smape(np.expm1(y_log), np.expm1(ridge_oof)))

# ----------------------------
# Tabular features
# ----------------------------
print('Engineeering tabular features...')
train['ipq'] = train['catalog_content'].apply(extract_ipq)
test['ipq'] = test['catalog_content'].apply(extract_ipq)

train['brand'] = train['catalog_content'].apply(safe_brand)
test['brand'] = test['catalog_content'].apply(safe_brand)

# consistent brand encoding
brands = pd.concat([train['brand'], test['brand']]).astype('category')
brand_map = {c:i for i,c in enumerate(brands.cat.categories)}
train['brand_enc'] = train['brand'].map(brand_map).fillna(-1).astype(int)
test['brand_enc'] = test['brand'].map(brand_map).fillna(-1).astype(int)

train['text_len'] = train['catalog_content'].str.len()
test['text_len'] = test['catalog_content'].str.len()

# ----------------------------
# Download & cache images (only missing)
# ----------------------------
print('Downloading images (this may take a while)...')
# collect unique links and ensure consistent filenames
train_links = train['image_link'].fillna('').astype(str).tolist()
test_links = test['image_link'].fillna('').astype(str).tolist()
all_links = train_links + test_links

# Parallel download (adjust workers based on Colab runtime)
_download_workers = 12
parallel_download(all_links, savefolder=DOWNLOAD_FOLDER, workers=_download_workers)
print('Download pass completed.')

# %%
# ----------------------------
# VGG16 feature extraction (CPU-friendly)
# Caches per-image .npy files in FEATURE_CACHE
# ----------------------------
print('Preparing VGG16...')
device = 'cpu'
model = models.vgg16(pretrained=True)
# replace classifier to output penultimate features (4096)
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-1])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

os.makedirs(FEATURE_CACHE, exist_ok=True)

from functools import lru_cache

def image_path_to_cache(url):
    name = Path(url).name.split('?')[0]
    return os.path.join(FEATURE_CACHE, name + '.npy')


def extract_and_cache_feature(url):
    cache_file = image_path_to_cache(url)
    if os.path.exists(cache_file):
        try:
            return np.load(cache_file)
        except:
            pass
    # find downloaded file
    fname = os.path.join(DOWNLOAD_FOLDER, Path(url).name.split('?')[0])
    if not os.path.exists(fname):
        # fallback: try to download single image
        fname = download_image(url, savefolder=DOWNLOAD_FOLDER)
        if fname is None:
            feat = np.zeros(4096, dtype=np.float32)
            np.save(cache_file, feat)
            return feat
    try:
        img = Image.open(fname).convert('RGB')
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = model(x).cpu().numpy().reshape(-1)
    except Exception as e:
        feat = np.zeros(4096, dtype=np.float32)
    try:
        np.save(cache_file, feat)
    except:
        pass
    return feat

# Batch extraction with multiprocessing
from multiprocessing import Pool

def batch_extract_features(urls, workers=6):
    with Pool(workers) as p:
        feats = list(tqdm(p.imap(extract_and_cache_feature, urls), total=len(urls)))
    return np.vstack(feats)

print('Extracting features for train images...')
start = time.time()
train_img_feats = batch_extract_features(train_links, workers=6)
print('Train images features shape:', train_img_feats.shape, 'time:', time.time()-start)

print('Extracting features for test images...')
start = time.time()
test_img_feats = batch_extract_features(test_links, workers=6)
print('Test images features shape:', test_img_feats.shape, 'time:', time.time()-start)

# ----------------------------
# PCA reduce image features
# ----------------------------
print('Running PCA on image features...')
pca = PCA(n_components=PCA_IMG_DIM, random_state=SEED)
train_img_pca = pca.fit_transform(train_img_feats)
test_img_pca = pca.transform(test_img_feats)
print('PCA shapes:', train_img_pca.shape, test_img_pca.shape)

# ----------------------------
# Build LightGBM dataset (tabular + image PCA)
# ----------------------------
print('Preparing LightGBM data...')
X_tab_train = np.vstack([train['ipq'].values, train['brand_enc'].values, train['text_len'].values]).T
X_tab_test = np.vstack([test['ipq'].values, test['brand_enc'].values, test['text_len'].values]).T

X_lgb_train = np.hstack([X_tab_train, train_img_pca])
X_lgb_test = np.hstack([X_tab_test, test_img_pca])

# ----------------------------
# Train LightGBM (KFold)
# ----------------------------
print('Training LightGBM on tabular+image features...')
oof_lgb = np.zeros(len(train))
preds_lgb = np.zeros(len(test))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_lgb_train), 1):
    print(f'Fold {fold}/{N_FOLDS}')
    Xtr, Xval = X_lgb_train[tr_idx], X_lgb_train[val_idx]
    ytr, yval = y_log[tr_idx], y_log[val_idx]
    model_lgb = lgb.LGBMRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs= -1,
        random_state=SEED
    )
    model_lgb.fit(Xtr, ytr, eval_set=[(Xval,yval)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
    oof_lgb[val_idx] = model_lgb.predict(Xval)
    preds_lgb += model_lgb.predict(X_lgb_test) / N_FOLDS

print('LightGBM OOF SMAPE:', smape(np.expm1(y_log), np.expm1(oof_lgb)))

# ----------------------------
# Ensemble: Ridge (text) + LightGBM (tab+image)
# ----------------------------
print('Ensembling predictions...')
# weights can be tuned; start with 0.6 text, 0.4 lgb
w_text = 0.6
w_lgb = 0.4
final_preds_log = w_text * ridge_preds_test + w_lgb * preds_lgb
final_preds = np.expm1(final_preds_log)
final_preds = np.clip(final_preds, 0.01, None)

# ----------------------------
# Save submission
# ----------------------------
sub = pd.DataFrame({'sample_id': test['sample_id'], 'price': final_preds})
out_path = 'test_out.csv'
sub.to_csv(out_path, index=False, float_format='%.4f')
print('Saved', out_path)
print(sub.head())

# ----------------------------
# End
# ----------------------------
print('Done')
