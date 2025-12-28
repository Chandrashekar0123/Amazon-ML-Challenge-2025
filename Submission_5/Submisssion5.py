# -*- coding: utf-8 -*-
"""
High-Performance LSTM Training Script for Amazon Product Prices
Includes numeric features (Value, Unit) and catalog text.
"""

# =========================
# Mount Drive
# =========================
from google.colab import drive
drive.mount('/content/drive')

# =========================
# Imports
# =========================
import os
import re
import numpy as np
import pandas as pd
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================
# CONFIG
# =========================
DATASET_FOLDER = '/content/drive/MyDrive/Amazon_ML/student_resource/dataset/'
MAX_NUM_WORDS = 30000
MAX_SEQ_LEN = 120
EMBED_DIM = 128
BATCH_SIZE = 64
EPOCHS = 30

# =========================
# Load Data
# =========================
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))

# =========================
# Text Cleaning Function
# =========================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                # Remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # Remove URLs
    text = re.sub(r"[^\w\s]", " ", text)             # Remove special characters / emojis
    text = re.sub(r"\d+oz", " ounce ", text)         # Normalize ounces
    text = re.sub(r"\d+fl\s*oz", " ounce ", text)
    text = re.sub(r"\d+lb", " pound ", text)
    text = re.sub(r"\s+", " ", text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text.strip()

train['clean_text'] = train['catalog_content'].fillna("").apply(clean_text)

# =========================
# Numeric Feature Extraction
# =========================
# Safe Value extraction
def extract_value(text):
    try:
        match = re.search(r"Value[:\s]*([\d\.]+)", str(text))
        if match:
            val_str = match.group(1)
            # Only convert valid numeric strings
            if val_str.replace('.', '', 1).isdigit():
                return float(val_str)
        return 1.0  # default if missing or invalid
    except:
        return 1.0

train['Value_feat'] = train['catalog_content'].apply(extract_value)

# Safe Unit extraction
def extract_unit(text):
    try:
        match = re.search(r"Unit[:\s]*([a-zA-Z]+)", str(text))
        if match:
            return match.group(1)
        return "NA"
    except:
        return "NA"

unit_list = train['catalog_content'].apply(extract_unit)
unit_encoder = LabelEncoder()
train['Unit_feat'] = unit_encoder.fit_transform(unit_list)

# =========================
# Tokenizer
# =========================
tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train['clean_text'])
X_text_seq = tokenizer.texts_to_sequences(train['clean_text'])
X_text_pad = pad_sequences(X_text_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

# Combine numeric features
X_numeric = train[['Value_feat', 'Unit_feat']].values
y = train['price'].values

# Train-validation split
X_text_tr, X_text_val, X_num_tr, X_num_val, y_tr, y_val = train_test_split(
    X_text_pad, X_numeric, y, test_size=0.15, random_state=42
)

# =========================
# Build LSTM Model with Numeric Features
# =========================
# Text Input branch
text_input = Input(shape=(MAX_SEQ_LEN,))
x = Embedding(input_dim=MAX_NUM_WORDS, output_dim=EMBED_DIM, input_length=MAX_SEQ_LEN)(text_input)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
x = BatchNormalization()(x)
x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.1))(x)
x = BatchNormalization()(x)

# Numeric input branch
num_input = Input(shape=(2,))  # Value + Unit

# Combine
combined = Concatenate()([x, num_input])
z = Dense(128, activation='relu')(combined)
z = Dropout(0.3)(z)
z = Dense(64, activation='relu')(z)
z = Dropout(0.2)(z)
output = Dense(1, activation='linear')(z)

model = Model(inputs=[text_input, num_input], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mean_absolute_error')
model.summary()

# =========================
# Callbacks
# =========================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# =========================
# Train Model
# =========================
history = model.fit(
    [X_text_tr, X_num_tr], y_tr,
    validation_data=([X_text_val, X_num_val], y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[reduce_lr, early_stop]
)

# =========================
# Evaluate Model
# =========================
y_val_pred = model.predict([X_text_val, X_num_val]).flatten()
mae_val = mean_absolute_error(y_val, y_val_pred)
print(f"Validation MAE: {mae_val:.2f}")

# =========================
# Save Model, Tokenizer & Unit Encoder
# =========================
os.makedirs('models', exist_ok=True)
model.save('models/lstm_model.h5')
with open('models/tokenizer_lstm.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('models/unit_encoder.pkl', 'wb') as f:
    pickle.dump(unit_encoder, f)

print("✅ Model, tokenizer, and unit encoder saved successfully!")


# As per required formatt

# -*- coding: utf-8 -*-
"""
High-Performance LSTM Predictor for Amazon Product Prices
Uses catalog_content + numeric features (Value, Unit)
"""

import os
import re
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# =========================
# Text Preprocessing
# =========================
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

MAX_SEQ_LEN = 120  # must match training

def clean_text(text):
    """
    Clean and preprocess catalog_content
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)                # remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^\w\s]", " ", text)             # remove emojis/special chars
    text = re.sub(r"\d+oz", " ounce ", text)         # normalize ounces
    text = re.sub(r"\d+fl\s*oz", " ounce ", text)   
    text = re.sub(r"\d+lb", " pound ", text)         # normalize pounds
    text = re.sub(r"\s+", " ", text)                 # remove extra spaces
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text.strip()

# =========================
# Predictor Function
# =========================
def predictor(sample_id, catalog_content, image_link=None):
    """
    Predict price using pre-trained LSTM + numeric features model
    """
    global lstm_model, lstm_tokenizer, unit_encoder
    
    try:
        # Load model/tokenizer/unit encoder if not already loaded
        if 'lstm_model' not in globals():
            lstm_model = load_model('models/lstm_model.h5', compile=False)
            with open('models/tokenizer_lstm.pkl', 'rb') as f:
                lstm_tokenizer = pickle.load(f)
            with open('models/unit_encoder.pkl', 'rb') as f:
                unit_encoder = pickle.load(f)

        # -----------------
        # 1. Preprocess text
        # -----------------
        cleaned_text = clean_text(catalog_content)
        seq = lstm_tokenizer.texts_to_sequences([cleaned_text])
        padded = pad_sequences(seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

        # -----------------
        # 2. Extract numeric features safely
        # -----------------
        # Value
        value_match = re.search(r"Value[:\s]*([\d\.]+)", str(catalog_content))
        try:
            value = float(value_match.group(1)) if value_match and value_match.group(1).replace('.', '', 1).isdigit() else 1.0
        except:
            value = 1.0
        
        # Unit
        unit_match = re.search(r"Unit[:\s]*([a-zA-Z]+)", str(catalog_content))
        try:
            unit = unit_encoder.transform([unit_match.group(1)])[0] if unit_match else 0
        except:
            unit = 0

        numeric_features = np.array([[value, unit]])

        # -----------------
        # 3. Predict
        # -----------------
        price = lstm_model.predict([padded, numeric_features])[0][0]
        price = max(price, 0.0)  # ensure non-negative
        
        return price

    except Exception as e:
        print(f"Error predicting sample {sample_id}: {e}")
        return -1.0

# =========================
# Main Script
# =========================
if __name__ == "__main__":
    DATASET_FOLDER = 'dataset'
    
    # Load test dataset
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    # Apply predictor to each row
    test['price'] = test.apply(
        lambda row: predictor(row['sample_id'], row['catalog_content'], row['image_link']),
        axis=1
    )
    
    # Save only required columns
    output_df = test[['sample_id', 'price']]
    output_path = os.path.join(DATASET_FOLDER, 'test_out.csv')
    output_df.to_csv(output_path, index=False)
    
    print(f"Predictions saved to {output_path}")
    print("Sample predictions:\n", output_df.head())

