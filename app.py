import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import requests
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Data Analysis", page_icon=":star:", layout="wide")

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie = load_lottie('https://assets3.lottiefiles.com/packages/lf20_49rdyysj.json')


with st.container():
    st.subheader('Hey, I am Ashmit Mahindroo studying in Class 9th :wave:')
    st.write('---')
    left_column, right_column = st.columns(2)

    with left_column:
        st.title("This is the Analysis on ProjectBeta Bootcamp")
    with right_column:
        st_lottie(lottie, height=140)

data1 = pd.read_csv("./fifa_eda_stats.csv")

with st.container():
    st.header("1. Football Player Analysis")

    fig, ax = plt.subplots()
    ax.hist(data1['Age'], bins=8, color='skyblue', edgecolor='black')
    ax.set_xlabel('Age')
    ax.set_ylabel('Number of Players')
    ax.set_title('Histogram of Player Ages')
    st.pyplot(fig)
    players_above_21 = data1[data1['Age'] > 21].shape[0]
    st.write(f"### Ans 1) Number of players older than 21: {players_above_21}")

with st.container():
    nationality_counts = data1['Nationality'].value_counts()
    countries_of_interest = ['Germany', 'Argentina', 'England']
    other_count = nationality_counts[~nationality_counts.index.isin(countries_of_interest)].sum()
    
    pie_data1 = nationality_counts[countries_of_interest].tolist() + [other_count]
    pie_labels = countries_of_interest + ['Other']

    fig, ax = plt.subplots()
    ax.pie(pie_data1, labels=pie_labels, autopct='%1.1f%%', colors=['red', 'blue', 'green', 'orange'])
    ax.set_title('Percentage of Players by Nationality')
    st.pyplot(fig)
    st.write('Ans 2) Number of players from-')
    for country in countries_of_interest:
        st.write(f"{country}: {nationality_counts[country]}")
    st.write(f"Other countries: {other_count}")

st.write('---')
st.header("2. Linear Regression Model: Salary Prediction")

data3 = {
    'Years of Experience': [1.2, 1.4, 1.6, 2.1, 2.3, 3, 3.1, 3.3, 3.3, 3.8, 4, 4.1, 4.1, 4.2, 4.6, 5, 5.2, 5.4, 6, 6.1, 6.9, 7.2, 8, 8.3, 8.8, 9.1, 9.6, 9.7, 10.4, 10.6],
    'Salary': [39344, 46206, 37732, 43526, 39892, 56643, 60151, 54446, 64446, 57190, 63219, 55795, 56958, 57082, 61112, 67939, 66030, 83089, 81364, 93941, 91739, 98274, 101303, 113813, 109432, 105583, 116970, 112636, 122392, 121873]
}
df = pd.DataFrame(data3)

X = df[['Years of Experience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("Model Performance")
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue')
ax.plot(X, model.predict(X), color='red')
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')
st.pyplot(fig)

st.sidebar.header("Predict Salary")
years_experience = st.sidebar.number_input("Enter years of experience", min_value=0.0, step=0.1)
if years_experience:
    predicted_salary = model.predict(np.array([[years_experience]]))
    st.sidebar.write(f"Predicted Salary: ${predicted_salary[0]:,.2f}")
st.write('---')
st.header("3. AirBnb Location Analysis")

data2 = pd.read_csv("./airbnb.csv")

with st.container():
    
    location_counts = data2['address'].str.extract(r'(\bNew Delhi\b|\bMumbai\b|\bTurkey\b)', expand=False).fillna('Other').value_counts()
    
    total_listings = len(data2)
    percentages = (location_counts / total_listings) * 100
    
    fig, ax = plt.subplots()
    ax.bar(percentages.index, percentages.values, color=['blue', 'green', 'red', 'grey'])
    ax.set_xlabel('Location')
    ax.set_ylabel('Percentage')
    ax.set_title('Percentage of AirBnb Listings by Location')
    st.pyplot(fig)

    st.write("### Ans 1) Percentage of AirBnb Listings by Location")
    for location, percentage in percentages.items():
        st.write(f"{location}: {percentage:.2f}%")

with st.container():
    data2['rating'] = pd.to_numeric(data2['rating'], errors='coerce').fillna(0).astype(int)
    
    rating_counts = data2['rating'].value_counts().sort_index()
    
    rating_labels = {0: 'new', 3: '3', 4: '4', 5: '5'}
    rating_counts.index = rating_counts.index.map(rating_labels)
    
    fig, ax = plt.subplots()
    ax.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Set3.colors)
    ax.set_title('Percentage of AirBnb Listings by Rating')
    
    st.pyplot(fig)
    high_ratings = data2[data2['rating'] > 4]
    
    count_high_ratings = len(high_ratings)
    st.write('*floats are converted into int')
    st.write(f"### Ans 2) Number of listings with rating more than 4: {count_high_ratings}")

criteria1 = {'guests': 15, 'bedrooms': 1, 'beds': 2, 'bathrooms': 1}
criteria2 = {'guests': 2, 'bedrooms': 1, 'beds': 1, 'bathrooms': 1}

matches_criteria1 = data2[
    (data2['guests'] == criteria1['guests']) &
    (data2['bedrooms'] == criteria1['bedrooms']) &
    (data2['beds'] == criteria1['beds']) &
    (data2['bathrooms'] == criteria1['bathrooms'])
]

matches_criteria2 = data2[
    (data2['guests'] == criteria2['guests']) &
    (data2['bedrooms'] == criteria2['bedrooms']) &
    (data2['beds'] == criteria2['beds']) &
    (data2['bathrooms'] == criteria2['bathrooms'])
]

count_matches_criteria1 = len(matches_criteria1)
count_matches_criteria2 = len(matches_criteria2)

total_listings = len(data2)

others = total_listings - count_matches_criteria1 - count_matches_criteria2

labels = ['Criteria 1', 'Criteria 2', 'Others']
sizes = [count_matches_criteria1, count_matches_criteria2, others]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])
ax.set_title('Distribution of Airbnb Listings')

st.pyplot(fig)

st.header('Ans 3)')
st.write("Criteria 1: 15 guests, 1 bedroom, 2 beds, 1 bathroom")
st.write("Criteria 2: 2 guests, 1 bedroom, 1 bed, 1 bathroom")
st.write(f"Number of Airbnbs matching Criteria 1: {count_matches_criteria1}")
st.write(f"Number of Airbnbs matching Criteria 2: {count_matches_criteria2}")