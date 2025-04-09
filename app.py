import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


model_path = 'KNN.sav'

@st.cache_resource
def load_model():
    with open(model_path, 'rb') as f:
        return pickle.load(f)

model = load_model()

st.title('🔬 Breast Cancer Prediction App')

page = st.sidebar.radio("Select Feature", ["📊 Make Prediction", "📁 Retrain Model"])

# --- Prediction Page ---
if page == "📊 Make Prediction":
    st.subheader("🧪 Input Features")
    radius = st.number_input("Mean Radius", value=0.0)
    texture = st.number_input("Mean Texture", value=0.0)
    perimeter = st.number_input("Mean Perimeter", value=0.0)
    area = st.number_input("Mean Area", value=0.0)
    smoothness = st.number_input("Mean Smoothness", value=0.0)
    compactness = st.number_input("Mean Compactness", value=0.0)

    input_data = pd.DataFrame([[radius, texture, perimeter, area, smoothness, compactness]],
                              columns=['Mean Radius', 'Mean Texture', 'Mean Perimeter', 'Mean Area', 'Mean Smoothness', 'Mean Compactness'])

    if st.button('🚀 Predict'):
        try:
            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data).max() if hasattr(model, "predict_proba") else None

            label_mapping = {0: 'Benign', 1: 'Malignant'}
            result_label = label_mapping.get(prediction, "Unknown")

            explanation = "Benign tumors are usually not life-threatening." if prediction == 0 else \
                          "Malignant tumors are cancerous and need urgent attention."

            st.success(f"🟢 Prediction: {result_label}")
            if confidence:
                st.info(f"🎯 Confidence: {round(confidence * 100, 2)}%")
            st.markdown(f"📝 **Explanation:** {explanation}")

            input_data["Prediction"] = result_label
            if confidence:
                input_data["Confidence (%)"] = round(confidence * 100, 2)
            input_data["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report", data=csv, file_name="Prediction_Report.csv", mime="text/csv")

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            st.error(f"❌ Error making prediction: {e}")

# --- Retrain Page ---
elif page == "📁 Retrain Model":
    st.subheader("📤 Upload Data to Retrain the Model")
    uploaded_file = st.file_uploader("Upload CSV with new training data", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", df.head())

            if 'target' not in df.columns:
                st.error("❌ Uploaded file must contain a 'target' column.")
            else:
                X = df.drop(columns='target')
                y = df['target']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                from sklearn.neighbors import KNeighborsClassifier
                model_new = KNeighborsClassifier()
                model_new.fit(X_train, y_train)

                pickle.dump(model_new, open(model_path, 'wb'))

                st.success("✅ Model retrained and saved successfully!")

                y_pred = model_new.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                st.markdown(f"### 📈 Accuracy: {round(acc * 100, 2)}%")
                st.text("📋 Classification Report:\n" + report)

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_title("Confusion Matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            

        except Exception as e:
            logging.error(f"Retraining error: {e}")
            st.error(f"❌ Error retraining model: {e}")

