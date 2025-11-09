import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter
import time
import numpy as np


# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('snu_friendship.csv')
    columns = ['Hobby_top1', 'Hobby top2', 'Club top1', 'Club top 2', 'Teamwork preference  \n ']
    # Rename columns to match expected names
    column_mapping = {
        'Hobby_top1': 'Hobby_top1',
        'Hobby top2': 'Hobby top2',
        'Club top1': 'Club top1',
        'Club top 2': 'Club top 2',
        'Teamwork preference  \n ': 'Teamwork preference  \n '
    }
    df.rename(columns=column_mapping, inplace=True)
    df_selected = df[columns].dropna()
    return df_selected

# Train and save the model
@st.cache_data
def train_model(df_selected):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_data = encoder.fit_transform(df_selected)

    # Fit K-Means with optimal k=4
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(encoded_data)

    return encoder, kmeans, encoded_data, clusters

# Function to predict cluster
def predict_cluster(encoder, kmeans, hobby_top1, hobby_top2, club_top1, club_top2, teamwork_pref):
    new_data = pd.DataFrame({
        'Hobby_top1': [hobby_top1],
        'Hobby top2': [hobby_top2],
        'Club top1': [club_top1],
        'Club top 2': [club_top2],
        'Teamwork preference  \n ': [teamwork_pref]
    })
    encoded_new = encoder.transform(new_data)
    predicted_cluster = kmeans.predict(encoded_new)[0]
    return predicted_cluster

# Streamlit app
def main():
    st.title("🎓 The Friendship Blueprint of SNU - Find the club best suited to you")
    st.markdown("Discover which friendship group you belong to based on your hobbies and interests!")

    # Load data and train model
    df_selected = load_data()
    encoder, kmeans, encoded_data, clusters = train_model(df_selected)

    # Sidebar for inputs
    st.sidebar.header("📝 Enter Your Details")
    hobby_top1 = st.sidebar.selectbox("Top Hobby 1", df_selected['Hobby_top1'].unique())
    hobby_top2 = st.sidebar.selectbox("Top Hobby 2", df_selected['Hobby top2'].unique())
    club_top1 = st.sidebar.selectbox("Top Club 1", df_selected['Club top1'].unique())
    club_top2 = st.sidebar.selectbox("Top Club 2", df_selected['Club top 2'].unique())
    teamwork_pref = st.sidebar.slider("Teamwork Preference (1-5)", 1, 5, 3)

    if st.sidebar.button("🔮 Predict My Club"):
        predicted_cluster = predict_cluster(encoder, kmeans, hobby_top1, hobby_top2, club_top1, club_top2, teamwork_pref)

        # Club nicknames and descriptions
        club_nicknames = {
            0: "🦋 Social Butterflies",
            1: "🎨 Creative Minds",
            2: "⚽ Active Enthusiasts",
            3: "📚 Thoughtful Individuals"
        }

        club_descriptions = {
            0: "🌟 **Social Butterflies**: You love group activities and teamwork. Great for collaborative projects!",
            1: "🎨 **Creative Minds**: Artistic and hobby-focused. Perfect for creative clubs and events!",
            2: "⚽ **Active Enthusiasts**: Sports and outdoor hobbies. Ideal for sports teams and adventure groups!",
            3: "📚 **Thoughtful Individuals**: Prefer independent activities with some teamwork. Suited for study groups and quiet gatherings!"
        }

        nickname = club_nicknames.get(predicted_cluster, f"Club {predicted_cluster}")
        st.success(f"🎉 You should join **{nickname} Club**!")

        st.markdown(club_descriptions.get(predicted_cluster, "A unique club just for you!"))

        # Display predicted club with icon and message
        st.header("🎯 Your Predicted Club")

        # Club icons and names
        club_icons = {
            0: "🦋",
            1: "🎨",
            2: "⚽",
            3: "📚"
        }

        club_full_names = {
            0: "Social Butterflies",
            1: "Creative Minds",
            2: "Active Enthusiasts",
            3: "Thoughtful Individuals"
        }

        icon = club_icons.get(predicted_cluster, "🎉")
        full_name = club_full_names.get(predicted_cluster, f"Club {predicted_cluster}")

        # Large icon display
        st.markdown(f"<h1 style='text-align: center; font-size: 100px;'>{icon}</h1>", unsafe_allow_html=True)

        # Club name and message
        st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{full_name}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size: 18px;'>🎉 Congratulations! Based on your hobbies and interests, you are a perfect fit for the <strong>{full_name}</strong> club!</p>", unsafe_allow_html=True)

    # Remove the Club Visualization section

    # Club sizes and summaries
    st.header("🏛️ About Clubs")
    cluster_counts = Counter(clusters)
    df_selected['Cluster'] = clusters

    # Define club_nicknames here as well
    club_nicknames = {
        0: "🦋 Social Butterflies",
        1: "🎨 Creative Minds",
        2: "⚽ Active Enthusiasts",
        3: "📚 Thoughtful Individuals"
    }

    for cluster in sorted(cluster_counts.keys()):
        nickname = club_nicknames.get(cluster, f"Club {cluster}")
        st.subheader(f"{nickname} Club")

        # Get cluster data
        cluster_data = df_selected[df_selected['Cluster'] == cluster]

        # Most common hobbies
        top_hobby1 = cluster_data['Hobby_top1'].mode().iloc[0] if not cluster_data['Hobby_top1'].mode().empty else "N/A"
        top_hobby2 = cluster_data['Hobby top2'].mode().iloc[0] if not cluster_data['Hobby top2'].mode().empty else "N/A"

        # Most common clubs
        top_club1 = cluster_data['Club top1'].mode().iloc[0] if not cluster_data['Club top1'].mode().empty else "N/A"
        top_club2 = cluster_data['Club top 2'].mode().iloc[0] if not cluster_data['Club top 2'].mode().empty else "N/A"

        # Average teamwork preference
        avg_teamwork = cluster_data['Teamwork preference  \n '].mean()

        st.write(f"**Top Hobbies:** {top_hobby1}, {top_hobby2}")
        st.write(f"**Top Clubs:** {top_club1}, {top_club2}")
        st.write(f"**Average Teamwork Preference:** {avg_teamwork:.2f}")
        st.write("---")

    # Visualizations Section
    st.header("📊 Visualizations")

    # Bar Chart of Cluster Sizes
    st.subheader("📈 Cluster Sizes")
    fig, ax = plt.subplots(figsize=(8, 5))
    cluster_sizes = [cluster_counts[cluster] for cluster in sorted(cluster_counts.keys())]
    club_names = [club_nicknames.get(cluster, f"Club {cluster}") for cluster in sorted(cluster_counts.keys())]
    colors = plt.cm.tab10.colors[:len(cluster_sizes)]
    ax.bar(club_names, cluster_sizes, color=colors)
    ax.set_xlabel('Clubs')
    ax.set_ylabel('Number of Students')
    ax.set_title('Number of Students per Club')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Pie Chart of Cluster Distribution
    st.subheader("🥧 Cluster Distribution")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(cluster_sizes, labels=club_names, autopct='%1.1f%%', startangle=140, colors=colors)
    ax.set_title('Distribution of Students Across Clubs')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # PCA Scatter Plot
    st.subheader("🔍 PCA Scatter Plot of Clusters")
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(encoded_data)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title('2D PCA Visualization of Student Clusters')
    ax.grid(True, linestyle='--', alpha=0.5)
    # Create custom legend with club names
    unique_clusters = sorted(set(clusters))
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.tab10(i), markersize=10, label=club_nicknames.get(i, f"Club {i}")) for i in unique_clusters]
    ax.legend(handles=legend_elements, title="Clubs")
    st.pyplot(fig)

    # Bar Charts for Top Hobbies per Cluster
    st.subheader("🎨 Top Hobbies per Cluster")
    for cluster in sorted(cluster_counts.keys()):
        nickname = club_nicknames.get(cluster, f"Club {cluster}")
        st.markdown(f"**{nickname}**")
        cluster_data = df_selected[df_selected['Cluster'] == cluster]
        all_hobbies = pd.concat([cluster_data['Hobby_top1'], cluster_data['Hobby top2']])
        hobby_counts = all_hobbies.value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8, 4))
        hobby_counts.plot(kind='bar', ax=ax, color=plt.cm.tab10.colors[cluster % 10])
        ax.set_xlabel('Hobbies')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Top 5 Hobbies in {nickname}')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Machine Learning")

if __name__ == "__main__":
    main()
