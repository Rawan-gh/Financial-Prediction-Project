import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load the pre-trained model
pipe = pickle.load(open("finalised_model.pickle", 'rb'))

# Load the dataset (replace with your actual dataset path or data)
df = pd.read_csv('SP500_Stock_Data.csv')

# Apply custom styling with HTML and Markdown using blue shades
st.markdown(
    """
    <style>
    .fun-title {
        color: #1E90FF; /* Dodger Blue */
        font-size: 48px;
        text-align: center;
        font-weight: bold;
    }
    .fun-subtitle {
        color: #4169E1; /* Royal Blue */
        font-size: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .data-table-title {
        color: #1E90FF; /* Dodger Blue */
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .scatter-title {
        color: #1E90FF; /* Dodger Blue */
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .scatter-description {
        color: #00BFFF; /* Deep Sky Blue */
        font-size: 18px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Title of the app with blue shade
st.markdown('<div class="fun-title">Financial Prediction</div>', unsafe_allow_html=True)

# Subtitle with names on separate lines and blue shade
st.markdown(
    '<div class="fun-subtitle">This project is created by:<br>Rawan Alghannam<br>Hadeel Alghassab<br>Yara Alsardi</div>',
    unsafe_allow_html=True
)

# Create input fields for Interest Rate and Employment
interest_rate = st.text_input("Enter Interest Rate:", placeholder="e.g. 2.5")
employment = st.text_input("Enter Employment Rate:", placeholder="e.g. 70")

# Place the Predict button directly under the input fields
if st.button("Predict"):
    # Ensure both inputs are provided
    if interest_rate == "" or employment == "":
        st.error("Please enter both Interest Rate and Employment Rate.")
    else:
        try:
            # Ensure that input values are in expected ranges
            if float(interest_rate) < 0 or float(employment) < 0:
                st.error("Invalid input: Values must be non-negative.")
            else:
                # Predict based on the provided input
                prediction = pipe.predict(pd.DataFrame([[interest_rate, employment]]))[0]
                st.success(f"Prediction Result: {round(prediction, 2)}")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Display the table of data with a blue-shaded header
st.markdown('<div class="data-table-title">Data Table</div>', unsafe_allow_html=True)
st.markdown('<div class="fun-subtitle">Let’s take a look at the data we\'re using for predictions:</div>', unsafe_allow_html=True)

# Display the data as a table
st.dataframe(df)

# Display the graph of Interest Rate vs Employment with blue shades for the headers
st.markdown('<div class="scatter-title">Data Visualization</div>', unsafe_allow_html=True)
st.markdown('<div class="scatter-description">Here’s a visual representation of Interest Rates vs Employment:</div>', unsafe_allow_html=True)

# Create a scatter plot for Interest Rate vs Employment
plt.figure(figsize=(10, 6))
plt.scatter(df['Interest Rates'], df['Employment'], color='#4682B4')  # Steel Blue for points
plt.title('Interest Rates vs Employment', fontsize=16, color='#1E90FF')  # Dodger Blue title
plt.xlabel('Interest Rates', fontsize=12, color='#4169E1')  # Royal Blue for x-axis
plt.ylabel('Employment', fontsize=12, color='#4169E1')  # Royal Blue for y-axis
st.pyplot(plt)
