import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load dataset
data = pd.read_csv('creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data.Class == 0]
fraud = data[data.Class == 1]

# Undersample legitimate transactions to balance the classes
legit_sample = legit.sample(n=len(fraud), random_state=2)
data = pd.concat([legit_sample, fraud], axis=0)

# Split features and labels
X = data.drop(columns='Class')
Y = data['Class']

# Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train logistic regression on full feature set
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy
train_acc = accuracy_score(model.predict(X_train), Y_train)
test_acc = accuracy_score(model.predict(X_test), Y_test)

# Streamlit App
st.title("Credit Card Fraud Detection (Logistic Regression)")
st.write(f"🔵 **Training Accuracy**: {train_acc:.4f}")
st.write(f"🟠 **Testing Accuracy**: {test_acc:.4f}")

# --- Input Fields for Prediction ---
st.subheader("🔍 Predict a Transaction")
input_features = st.text_input("Enter all 30 features separated by commas (e.g., V1,V2,...,Amount):")
if st.button("Predict"):
    try:
        input_values = np.array(input_features.split(','), dtype=np.float64)
        if input_values.shape[0] != 30:
            st.error("❌ Please enter exactly 30 values.")
        else:
            prediction = model.predict(input_values.reshape(1, -1))
            if prediction[0] == 0:
                st.success("✅ Legitimate Transaction")
            else:
                st.warning("⚠️ Fraudulent Transaction")
    except:
        st.error("❌ Invalid input. Make sure all values are numeric.")

# --- Plotting Decision Boundary using only 2 features (V1 and V2) ---
st.subheader("📈 Logistic Regression Decision Boundary (Using V1 & V2 Only)")

# Use only V1 and V2 for visualization
X_vis = data[['V1', 'V2']].values
y_vis = data['Class'].values

# Train new logistic regression on V1 and V2 only
model_vis = LogisticRegression()
model_vis.fit(X_vis, y_vis)

# Create meshgrid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model_vis.predict(grid).reshape(xx.shape)

# Plot with matplotlib
fig, ax = plt.subplots(figsize=(10, 6))
ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
ax.scatter(X_vis[y_vis == 0][:, 0], X_vis[y_vis == 0][:, 1], c='green', label='Legit', edgecolor='k', s=20)
ax.scatter(X_vis[y_vis == 1][:, 0], X_vis[y_vis == 1][:, 1], c='red', label='Fraud', edgecolor='k', s=20)
ax.set_xlabel("V1")
ax.set_ylabel("V2")
ax.set_title("Decision Boundary (Logistic Regression on V1 vs V2)")
ax.legend()
st.pyplot(fig)
plt.show()