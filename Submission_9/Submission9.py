import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau 
from tensorflow.keras.regularizers import l2 

# --- Constants ---
TRAIN_PATH = '/content/drive/MyDrive/Amazon_ML/student_resource/dataset/train.csv'
TEST_PATH = '/content/drive/MyDrive/Amazon_ML/student_resource/dataset/test.csv'
FEATURE_FOLDER = '/content/drive/MyDrive/Amazon_ML/student_resource/features'
MAX_NUM_WORDS = 40000 # Increased vocab size
MAX_SEQ_LEN = 200 # Increased sequence length
EMBEDDING_DIM = 128
# IMAGE_FEATURE_DIM = 512 # Removed: Image feature is now DISABLED for robustness
EPSILON = 1e-6 
MAX_EPOCHS = 100 
L2_REG = 1e-4 # Retained L2 Regularization

# --- Utility Functions ---

def clean_text(text):
    """Performs standard NLP cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"<.*?>", " ", text) 
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) 
    text = re.sub(r"[^a-z0-9\s\+\-x%./]", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_ipq(text):
    """Extracts Item Pack Quantity (IPQ) using regex."""
    if not isinstance(text, str):
        return 1

    patterns = [
        r'(\d+)\s*pack', r'pack of\s*(\d+)', r'(\d+)\s*ct\b',
        r'(\d+)\s*count\b', r'(\d+)\s*pcs\b', r'x\s*(\d+)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return max(1, int(match.group(1))) 
            except ValueError:
                continue

    return 1 

def smape_metric(y_true, y_pred):
    """Calculates SMAPE (Symmetric Mean Absolute Percentage Error) - Non-differentiable."""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_pred) + np.abs(y_true)) / 2.0
    return np.mean(numerator / np.maximum(denominator, EPSILON)) * 100

def custom_smape_loss(y_true, y_pred):
    """Keras custom loss function for SMAPE, operating on exponential values."""
    
    y_true_raw = K.exp(y_true) - EPSILON
    y_pred_raw = K.exp(y_pred) - EPSILON
    
    numerator = K.abs(y_pred_raw - y_true_raw)
    denominator = (K.abs(y_true_raw) + K.abs(y_pred_raw)) / 2.0
    
    return K.mean(numerator / K.maximum(denominator, K.epsilon()), axis=-1)

# IMAGE LOADING FUNCTION IS REMOVED TO PREVENT MOCKING NOISE FROM AFFECTING THE MODEL
# def download_and_get_image_features(...): 
#    ...

def build_text_ipq_model(input_dim_ipq, max_words, max_seq_len, embedding_dim):
    """Builds the Keras Functional API model combining Text and IPQ features only."""
    
    # 1. Text Branch (Deeper Convolutional Neural Network for Text)
    text_input = Input(shape=(max_seq_len,), name='text_input')
    text_emb = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_seq_len)(text_input)
    text_conv1 = Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(L2_REG))(text_emb) # Wider filter
    text_conv2 = Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(L2_REG))(text_conv1) 
    text_pool = GlobalMaxPooling1D()(text_conv2)
    text_dense_a = Dense(256, activation='relu', kernel_regularizer=l2(L2_REG))(text_pool) # Increased width
    text_drop = Dropout(0.4)(text_dense_a) # Increased dropout
    text_dense = Dense(128, activation='relu')(text_drop)

    # 2. IPQ (Structural) Branch - Increased width
    ipq_input = Input(shape=(input_dim_ipq,), name='ipq_input')
    ipq_dense = Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(ipq_input)
    ipq_drop = Dropout(0.2)(ipq_dense)
    ipq_dense_b = Dense(32, activation='relu')(ipq_drop)

    # 3. Combine Branches
    combined = Concatenate()([text_dense, ipq_dense_b])
    
    # 4. Final Regression Head - Deepest Head for Fusion
    dense_final = Dense(512, activation='relu', kernel_regularizer=l2(L2_REG))(combined)
    dense_final = Dropout(0.3)(dense_final)
    dense_final_b = Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(dense_final)
    dense_final_b = Dropout(0.2)(dense_final_b)
    output = Dense(1, activation='linear')(dense_final_b) # Predicts log_price

    model = Model(inputs=[text_input, ipq_input], outputs=output)
    
    model.compile(optimizer='adam', loss=custom_smape_loss)
    
    return model

# --- Main Execution ---

def main():
    
    # Load Data
    try:
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
    except Exception as e:
        print(f"Error loading data. Ensure paths are correct and files exist. Error: {e}")
        return

    # --- 1. Feature Engineering ---
    
    # Target Transformation: Log(price) for better model convergence
    train['log_price'] = np.log(train['price'].values + EPSILON)
    y_train_log = train['log_price'].values

    # Text Cleaning and IPQ Extraction
    print("1. Cleaning Text and Extracting IPQ...")
    for df in [train, test]:
        df['catalog_content'] = df['catalog_content'].fillna("")
        df['clean_text'] = df['catalog_content'].apply(clean_text)
        df['ipq'] = df['catalog_content'].apply(extract_ipq)

    # IPQ Scaling (Structural Feature)
    ipq_scaler = StandardScaler()
    X_train_ipq = ipq_scaler.fit_transform(train[['ipq']])
    X_test_ipq = ipq_scaler.transform(test[['ipq']])
    
    # IMAGE FEATURE LOADING IS SKIPPED TO PREVENT NOISE (Image features are not included in this model)
    print("2. WARNING: Image features have been intentionally DISABLED for test robustness.")

    # --- 2. NLP Processing ---
    
    print("3. Tokenizing and Padding Text...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train['clean_text'])

    X_train_seq = tokenizer.texts_to_sequences(train['clean_text'])
    X_test_seq = tokenizer.texts_to_sequences(test['clean_text'])

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    # --- 3. Train/Validation Split ---
    
    X_tr_text, X_val_text, y_tr, y_val = train_test_split(X_train_pad, y_train_log, test_size=0.1, random_state=42)
    X_tr_ipq, X_val_ipq, _, _ = train_test_split(X_train_ipq, y_train_log, test_size=0.1, random_state=42)
    
    # --- 4. Model Building and Training ---
    
    print("4. Building and Training Text+IPQ Model with Custom SMAPE Loss...")
    model = build_text_ipq_model(X_train_ipq.shape[1], MAX_NUM_WORDS, MAX_SEQ_LEN, EMBEDDING_DIM)
    
    # Define Callbacks for robust optimization
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True, 
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )
    callbacks_list = [early_stopping, reduce_lr]
    
    history = model.fit(
        # ONLY Text and IPQ inputs
        {'text_input': X_tr_text, 'ipq_input': X_tr_ipq}, y_tr,
        validation_data=({'text_input': X_val_text, 'ipq_input': X_val_ipq}, y_val),
        epochs=MAX_EPOCHS, 
        batch_size=128,
        callbacks=callbacks_list, 
        verbose=1
    )

    # --- 5. Validation Check ---
    
    # Predict on validation set (log price)
    y_val_pred_log = model.predict({'text_input': X_val_text, 'ipq_input': X_val_ipq}).flatten()
    
    # Inverse transform to raw price
    y_val_true_raw = np.exp(y_val) - EPSILON
    y_val_pred_raw = np.exp(y_val_pred_log) - EPSILON
    y_val_pred_raw = np.maximum(y_val_pred_raw, 0.01)
    
    # Calculate SMAPE
    val_smape = smape_metric(y_val_true_raw, y_val_pred_raw)
    print(f"\n✅ Validation SMAPE (on {len(y_val)} samples): {val_smape:.4f}%")
    
    # --- 5.5. Full Training Data Check (All Data) ---
    print("\n📊 Calculating SMAPE on FULL Training Dataset (75k samples)...")
    
    # Predict on entire training set (log price)
    y_full_train_pred_log = model.predict({
        'text_input': X_train_pad, 
        'ipq_input': X_train_ipq
    }).flatten()
    
    # Inverse transform to raw price
    y_full_train_true_raw = np.exp(y_train_log) - EPSILON
    y_full_train_pred_raw = np.exp(y_full_train_pred_log) - EPSILON
    y_full_train_pred_raw = np.maximum(y_full_train_pred_raw, 0.01)
    
    # Calculate SMAPE
    full_train_smape = smape_metric(y_full_train_true_raw, y_full_train_pred_raw)
    print(f"✅ FULL Training SMAPE (All Data): {full_train_smape:.4f}%")
    
    # --- 6. Final Prediction and Submission ---

    print("5. Generating Final Test Predictions...")
    # Predict on the actual test set (log price) - ONLY Text and IPQ inputs
    test_preds_log = model.predict({'text_input': X_test_pad, 'ipq_input': X_test_ipq}).flatten()

    # Inverse Transform
    test_preds = np.exp(test_preds_log) - EPSILON
    
    # Constraint: Predicted prices must be positive float values.
    test_preds = np.maximum(test_preds, 0.01) 

    submission = pd.DataFrame({
        "sample_id": test['sample_id'],
        "price": test_preds
    })

    submission.to_csv("test_out.csv", index=False)
    print("\n📦 Submission file 'test_out.csv' created successfully!")

if __name__ == '__main__':
    # Increase precision for printing
    np.set_printoptions(precision=4) 
    
    main()
