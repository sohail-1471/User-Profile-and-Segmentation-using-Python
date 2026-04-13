# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("D:/Data Science/Data Science projects/Data analysis/Users Profile and segmentation/user_profiles_for_ads.csv")
pd.set_option('display.max_columns', None)
print(data.head())
print(data.isnull().sum())

# Exploratory Data Analysis

# setting the aesthetic style of the plots
sns.set_style("darkgrid")

# creating subplots for the demographic distributions
fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle('Distribution of Key Demographic Variables')


# age distribution
sns.countplot(ax=axes[0, 0], x='Age', data=data, hue="Age", palette="RdBu", legend=False)
axes[0, 0].set_title('Age Distribution')
axes[0, 0].tick_params(axis='x', rotation=45)

# gender distribution
sns.countplot(ax=axes[0, 1], x='Gender', data=data, hue="Gender", palette="RdBu", legend=False)
axes[0, 1].set_title('Gender Distribution')

# education level distribution
sns.countplot(ax=axes[1, 0], x='Education Level', data=data, hue="Education Level", palette="RdBu", legend=False)
axes[1, 0].set_title('Education Level Distribution')
axes[1, 0].tick_params(axis='x', rotation=45)

# income level distribution
sns.countplot(ax=axes[1, 1], x='Income Level', data=data, hue="Income Level", palette="RdBu", legend=False)
axes[1, 1].set_title('Income Level Distribution')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=6)
plt.show()

# device usage distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Device Usage', data=data, hue="Age", palette='summer')
plt.title('Device Usage Distribution')
plt.show()

# creating subplots for user online behavior and ad interaction metrics
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('User Online Behavior and Ad Interaction Metrics')

# time spent online on weekdays
sns.histplot(ax=axes[0, 0], x='Time Spent Online (hrs/weekday)', data=data, bins=20, kde=True, color='skyblue')
axes[0, 0].set_title('Time Spent Online on Weekdays')

# time spent online on weekends
sns.histplot(ax=axes[0, 1], x='Time Spent Online (hrs/weekend)', data=data, bins=20, kde=True, color='orange')
axes[0, 1].set_title('Time Spent Online on Weekends')

# likes and reactions
sns.histplot(ax=axes[1, 0], x='Likes and Reactions', data=data, bins=20, kde=True, color='green')
axes[1, 0].set_title('Likes and Reactions')

# click-through rates
sns.histplot(ax=axes[1, 1], x='Click-Through Rates (CTR)', data=data, bins=20, kde=True, color='red')
axes[1, 1].set_title('Click-Through Rates (CTR)')

# conversion rates
sns.histplot(ax=axes[2, 0], x='Conversion Rates', data=data, bins=20, kde=True, color='purple')
axes[2, 0].set_title('Conversion Rates')

# ad interaction time
sns.histplot(ax=axes[2, 1], x='Ad Interaction Time (sec)', data=data, bins=20, kde=True, color='brown')
axes[2, 1].set_title('Ad Interaction Time (sec)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=6)
plt.show()

from collections import Counter

# Splitting the 'Top Interests' column and creating a list of all interests
interests_list = data['Top Interests'].str.split(', ').sum()

# Counting the frequency of each interest
interests_counter = Counter(interests_list)

# converting the counter object to a DataFrame for easier plotting
interests_df = pd.DataFrame(interests_counter.items(), columns=['Interest', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Plotting the most common interest
plt.figure(figsize=(12,8))
sns.barplot(x='Frequency', y='Interest', data=interests_df.head(10), palette='RdBu')
plt.title('Top 10 Users Interests')
plt.xlabel('Frequency')
plt.ylabel('Interest')
plt.show()

# User Profiling and Segmentation
# We can now segment users into distinct groups for targeted ad campaigns.
# Segmentation can be based on various criteria, such as:
#
# Demographics: Age, Gender, Income Level, Education Level
# Behavioural: Time Spent Online, Likes and Reactions, CTR, Conversion Rates
# Interests: Aligning ad content with the top interests identified

# To implement user profiling and segmentation, we can apply clustering techniques or develop personas based on
# the combination of these attributes. This approach enables the creation of more personalized and effective ad
# campaigns, ultimately enhancing user engagement and conversion rates.
#
# Let’s start by selecting a subset of features that could be most indicative of user preferences and behaviour
# for segmentation and apply a clustering algorithm to create user segments:

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Selecting features for clustering
features = ['Age','Gender','Income Level','Time Spent Online (hrs/weekday)','Time Spent Online (hrs/weekend)','Likes and Reactions','Click-Through Rates (CTR)']

# Separating the features we want to consider for clustering
x = data[features]

# Defining preprocessing for numerical and categorical features
numeric_features = ['Time Spent Online (hrs/weekday)','Time Spent Online (hrs/weekend)','Likes and Reactions','Click-Through Rates (CTR)']
numeric_transformer = StandardScaler()

categorical_features = ['Age','Gender','Income Level']
categorical_transformer = OneHotEncoder()

# Combining preprocessing steps
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Creating a preprocessing and clustering pipline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('cluster', KMeans(n_clusters=5, random_state=42))])
pipeline.fit(x)
cluster_labels = pipeline.named_steps['cluster'].labels_
data['Cluster'] = cluster_labels
print(data.head())

# The clustering process has successfully segmented our users into five distinct groups (Clusters 0 to 4).
# Each cluster represents a unique combination of the features we selected, including age, gender, income level,
# online behaviour, and engagement metrics. These clusters can serve as the basis for creating targeted
# ad campaigns tailored to the preferences and behaviours of each segment.
#
# We’ll compute the mean values of the numerical features and the mode for categorical features within each
# cluster to get a sense of their defining characteristics:
# computing the mean values of numerical features for each cluster
cluster_means = data.groupby('Cluster')[numeric_features].mean()

for feature in categorical_features:
    mode_series = data.groupby('Cluster')[feature].agg(lambda x: x.mode()[0])
    cluster_means[feature] = mode_series

print(cluster_means)

# Now, we’ll assign each cluster a name that reflects its most defining characteristics based on the mean values
# of numerical features and the most frequent categories for categorical features. Based on the cluster analysis,
# we can summarize and name the segments as follows:
#
# Cluster 0 – “Weekend Warriors”: High weekend online activity, moderate likes and reactions, predominantly male,
#              age group 25-34, income level 80k-100k.
# Cluster 1 – “Engaged Professionals”: Balanced online activity, high likes and reactions, predominantly male,
#              age group 25-34, high income (100k+).
# Cluster 2 – “Low-Key Users”: Moderate to high weekend online activity, moderate likes and reactions,
#              predominantly male, age group 25-34, income level 60k-80k, lower CTR.
# Cluster 3 – “Active Explorers”: High overall online activity, lower likes and reactions, predominantly female,
#              age group 25-34, income level 60k-80k.
# Cluster 4 – “Budget Browsers”: Moderate online activity, lowest likes and reactions, predominantly female,
# age group 25-34, lowest income level (0-20k), lower CTR.

# preparing data for radar chart
features_to_plot = ['Time Spent Online (hrs/weekday)', 'Time Spent Online (hrs/weekend)', 'Likes and Reactions', 'Click-Through Rates (CTR)']
labels = np.array(features_to_plot)

# creating a dataframe for the radar chart
radar_df = cluster_means[features_to_plot].reset_index()

# normalizing the data
radar_df_normalized = radar_df.copy()
for feature in features_to_plot:
    radar_df_normalized[feature] = (radar_df[feature] - radar_df[feature].min()) / (radar_df[feature].max() - radar_df[feature].min())

# adding a full circle for plotting
radar_df_normalized = radar_df_normalized._append(radar_df_normalized.iloc[0])


# assigning names to segments
segment_names = ['Weekend Warriors', 'Engaged Professionals', 'Low-Key Users', 'Active Explorers', 'Budget Browsers']

import plotly.graph_objects as go
fig = go.Figure()

# loop through each segment to add to the radar chart
for i, segment in enumerate(segment_names):
    fig.add_trace(go.Scatterpolar(
        r=radar_df_normalized.iloc[i][features_to_plot].values.tolist() + [radar_df_normalized.iloc[i][features_to_plot].values[0]],  # Add the first value at the end to close the radar chart
        theta=labels.tolist() + [labels[0]],  # add the first label at the end to close the radar chart
        fill='toself',
        name=segment,
        hoverinfo='text',
        text=[f"{label}: {value:.2f}" for label, value in zip(features_to_plot, radar_df_normalized.iloc[i][features_to_plot])]+[f"{labels[0]}: {radar_df_normalized.iloc[i][features_to_plot][0]:.2f}"]  # Adding hover text for each feature
    ))

# update the layout to finalize the radar chart
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )),
    showlegend=True,
    title='User Segments Profile'
)

fig.show()


