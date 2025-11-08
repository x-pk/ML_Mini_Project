## 🎓 The Friendship Blueprint of SNU

Discover which friendship group you belong to based on your hobbies and interests! This project uses machine learning to cluster students into groups like Social Butterflies 🦋, Creative Minds 🎨, Active Enthusiasts ⚽, and Thoughtful Individuals 📚.

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Clustering Analysis** | K-Means clustering on student data with optimal k determination |
| 📊 **Data Visualization** | PCA-reduced 2D plots of clusters |
| 🎯 **Prediction App** | Interactive Streamlit web app for club recommendations |
| 📈 **Evaluation Metrics** | Silhouette Score and Davies-Bouldin Index for cluster quality |
| 🎨 **User-Friendly UI** | Emojis, large icons, and engaging messages |

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/x-pk/student-friendship-clustering.git
   cd student-friendship-clustering
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 📖 Usage

1. **Enter your details** in the sidebar: Top hobbies, clubs, and teamwork preference.
2. **Click "🔮 Predict My Club"** to get your recommendation.
3. **View your club** with a large emoji icon and personalized message.
4. **Explore "🏛️ About Clubs"** for summaries of each group.

## 🏗️ Project Structure

```
├── snu_friendship.csv          # Dataset
├── cluster_students.py         # Clustering script
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🔧 How It Works

1. **Data Preprocessing**: Load CSV, select columns, handle missing values, one-hot encode categoricals.
2. **Clustering**: Use K-Means (k=4) on encoded data.
3. **Evaluation**: Compute Silhouette Score and Davies-Bouldin Index.
4. **Visualization**: PCA for 2D cluster plots.
5. **Prediction**: Encode new inputs and predict cluster.
6. **Web App**: Interactive interface for user predictions.

## 📊 Evaluation Results

- **Optimal Clusters**: 4
- **Silhouette Score**: [Insert score from cluster_students.py]
- **Davies-Bouldin Index**: [Insert score from cluster_students.py]

## 🤝 Contributing

Feel free to fork and submit pull requests! For major changes, please open an issue first.


---
Pratik, Sohel, Manish
