import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_excel("Dataset_Scrapping.xlsx")

# Data Preprocessing
df.dropna(inplace=True)
X = df['full_text']
y = df['Sentiment']

# Text Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Save Model and Vectorizer
import joblib
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluate Model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy_percent = accuracy * 100
print("Akurasi: {:.2f}%".format(accuracy_percent))
