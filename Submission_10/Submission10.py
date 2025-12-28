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
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# --- Constants ---
TRAIN_PATH = '/content/drive/MyDrive/Amazon_ML/student_resource/dataset/train.csv'
TEST_PATH = '/content/drive/MyDrive/Amazon_ML/student_resource/dataset/test.csv'
MAX_NUM_WORDS = 40000 
MAX_SEQ_LEN = 200 
EMBEDDING_DIM = 256 # High capacity embedding
EPSILON = 1e-6 
MAX_EPOCHS = 100 
L2_REG = 1e-5 

# --- Utility Functions ---

def dataset_info(name, df):
    """Prints detailed information about a pandas DataFrame."""
    print(f"📊 Dataset: {name}")
    print("="*50)
    print("Shape (rows, columns):", df.shape)
    print("\nColumn Names:", df.columns.tolist())
    print("\nData Types:\n", df.dtypes)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nFirst 5 Rows:\n", df.head())
    print("\n" + "="*50 + "\n")

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

def build_text_ipq_model(input_dim_ipq, max_words, max_seq_len, embedding_dim):
    """Builds the Keras Functional API model combining Text and IPQ features only."""
    
    # 1. Text Branch (MAXIMUM CAPACITY Convolutional Neural Network for Text)
    text_input = Input(shape=(max_seq_len,), name='text_input')
    text_emb = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_seq_len)(text_input)
    
    text_conv1 = Conv1D(filters=512, kernel_size=3, activation='relu', kernel_regularizer=l2(L2_REG))(text_emb)
    text_conv2 = Conv1D(filters=256, kernel_size=5, activation='relu', kernel_regularizer=l2(L2_REG))(text_conv1) 
    text_pool = GlobalMaxPooling1D()(text_conv2)
    
    text_dense_a = Dense(512, activation='relu', kernel_regularizer=l2(L2_REG))(text_pool) 
    text_drop_a = Dropout(0.4)(text_dense_a) 
    # Output of Text branch, used for NN prediction AND XGBoost feature set
    text_features = Dense(128, activation='relu', name='text_features')(text_drop_a) 

    # 2. IPQ (Structural) Branch 
    ipq_input = Input(shape=(input_dim_ipq,), name='ipq_input')
    ipq_dense = Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(ipq_input)
    ipq_drop = Dropout(0.2)(ipq_dense)
    # Output of IPQ branch, used for NN prediction AND XGBoost feature set
    ipq_features = Dense(32, activation='relu', name='ipq_features')(ipq_drop)
    
    # 3. Combine Branches (Feature Fusion for NN)
    combined = Concatenate()([text_features, ipq_features])
    
    # 4. Final Regression Head (Deepest possible fusion layers)
    dense_final_a = Dense(1024, activation='relu', kernel_regularizer=l2(L2_REG))(combined)
    dense_final_a = Dropout(0.3)(dense_final_a)
    
    dense_final_b = Dense(512, activation='relu', kernel_regularizer=l2(L2_REG))(dense_final_a)
    dense_final_b = Dropout(0.2)(dense_final_b)
    
    dense_final_c = Dense(128, activation='relu', kernel_regularizer=l2(L2_REG))(dense_final_b)
    dense_final_c = Dropout(0.1)(dense_final_c)
    
    # Final output layer for the Keras NN model
    output = Dense(1, activation='linear', name='final_output')(dense_final_c) 

    model = Model(inputs=[text_input, ipq_input], outputs=[text_features, ipq_features, output])
    
    # FIX: Use a dictionary of losses, setting None for feature outputs we don't want to optimize
    model.compile(optimizer='adam', 
                  loss={'text_features': None, 'ipq_features': None, 'final_output': custom_smape_loss},
                  loss_weights={'text_features': 0, 'ipq_features': 0, 'final_output': 1.0}
                 )
    
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

    # --- Data Profiling ---
    dataset_info("Training Data", train)
    dataset_info("Test Data", test)

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
    
    print("2. WARNING: Image features have been intentionally DISABLED to maximize stability.")

    # --- 2. NLP Processing ---
    
    print("3. Tokenizing and Padding Text...")
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(train['clean_text'])

    # --- FIX: Tokenize text into sequences ---
    X_train_seq = tokenizer.texts_to_sequences(train['clean_text'])
    X_test_seq = tokenizer.texts_to_sequences(test['clean_text'])
    # ----------------------------------------
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')

    # --- 3. Train/Validation Split ---
    
    X_tr_text, X_val_text, y_tr, y_val = train_test_split(X_train_pad, y_train_log, test_size=0.1, random_state=42)
    X_tr_ipq, X_val_ipq, _, _ = train_test_split(X_train_ipq, y_train_log, test_size=0.1, random_state=42)
    
    # Keras requires the output structure to match during fit
    # We provide dummy outputs (y_zeros) for the feature layers
    y_zeros = np.zeros_like(y_tr)
    y_val_zeros = np.zeros_like(y_val)
    
    # --- 4. Model Building and Training ---
    
    print("4. Building and Training Text+IPQ Ensemble Components...")
    
    # Keras Model Setup
    model = build_text_ipq_model(X_train_ipq.shape[1], MAX_NUM_WORDS, MAX_SEQ_LEN, EMBEDDING_DIM)
    
    # Define Callbacks for robust optimization
    early_stopping = EarlyStopping(
        monitor='val_final_output_loss', # Monitor the loss of the actual price output
        patience=5, 
        restore_best_weights=True, 
        verbose=1,
        mode='min' # <<< FINAL FIX: Explicitly monitor for minimization
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_final_output_loss', # Monitor the loss of the actual price output
        factor=0.5, 
        patience=3, 
        min_lr=1e-6,
        verbose=1,
        mode='min' # <<< FINAL FIX: Explicitly monitor for minimization
    )
    callbacks_list = [early_stopping, reduce_lr]
    
    # Keras expects outputs for ALL three branches: text_features, ipq_features, final_output
    history = model.fit(
        {'text_input': X_tr_text, 'ipq_input': X_tr_ipq}, 
        {'text_features': y_zeros, 'ipq_features': y_zeros, 'final_output': y_tr},
        validation_data=({'text_input': X_val_text, 'ipq_input': X_val_ipq}, 
                         {'text_features': y_val_zeros, 'ipq_features': y_val_zeros, 'final_output': y_val}),
        epochs=MAX_EPOCHS, 
        batch_size=128,
        callbacks=callbacks_list, 
        verbose=1
    )

    # --- 5. Feature Extraction for XGBoost ---
    
    print("\n5. Feature Extraction and Ensemble Preparation...")
    
    # Extract learned features for XGBoost training
    # Keras Model.predict returns a list of outputs: [text_features, ipq_features, final_output]
    train_outputs = model.predict({'text_input': X_train_pad, 'ipq_input': X_train_ipq})
    test_outputs = model.predict({'text_input': X_test_pad, 'ipq_input': X_test_ipq})
    
    X_train_nn_text_features = train_outputs[0] # text_features (128 dim)
    X_train_nn_features = np.concatenate([
        X_train_nn_text_features,
        train[['ipq']].values # Use the raw IPQ value for XGBoost as well
    ], axis=1)

    X_test_nn_text_features = test_outputs[0]
    X_test_nn_features = np.concatenate([
        X_test_nn_text_features,
        test[['ipq']].values
    ], axis=1)

    # --- 6. XGBoost Training ---
    print("\n6. Training XGBoost Model...")
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.05, 
        'max_depth': 8,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(X_train_nn_features, label=y_train_log)
    
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=500, verbose_eval=False)

    # --- 7. Final Prediction and Ensemble ---
    print("\n7. Calculating Final Ensemble Predictions...")

    # Keras NN Prediction (log price) is the third output (index 2)
    nn_preds_log = test_outputs[2].flatten()

    # XGBoost Prediction (log price)
    dtest = xgb.DMatrix(X_test_nn_features)
    xgb_preds_log = xgb_model.predict(dtest)

    # Ensemble Weighted Average
    ENSEMBLE_WEIGHT_NN = 0.60
    ENSEMBLE_WEIGHT_XGB = 0.40
    
    ensemble_preds_log = (ENSEMBLE_WEIGHT_NN * nn_preds_log) + (ENSEMBLE_WEIGHT_XGB * xgb_preds_log)

    # Inverse Transform
    test_preds = np.exp(ensemble_preds_log) - EPSILON
    
    # Constraint: Predicted prices must be positive float values.
    test_preds = np.maximum(test_preds, 0.01) 

    submission = pd.DataFrame({
        "sample_id": test['sample_id'],
        "price": test_preds
    })

    submission.to_csv("test_out.csv", index=False)
    print("\n📦 FINAL ENSEMBLE Submission file 'test_out.csv' created successfully!")

if __name__ == '__main__':
    # Increase precision for printing
    np.set_printoptions(precision=4) 
    
    main()