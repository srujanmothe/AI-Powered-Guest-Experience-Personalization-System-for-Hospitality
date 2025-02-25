import streamlit as st
import pandas as pd
import numpy as np
import random
import xgboost
import joblib
import pymongo
from datetime import date
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient

client = MongoClient("mongodb+srv://srujanmothe62:bqfGoXX8dktIHyOp@cluster0.ew7nt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0&ssl=true")

# Initialize an empty DataFrame to store the data (you can later store it to a file or database)
columns = ['customer_id', 'Preferred Cuisine', 'age', 'check_in_date', 'check_out_date']
data = pd.DataFrame(columns=columns)

# Load Data (If provided, integrate dining_info.xlsx)
def load_data():
    try:
        df = pd.read_excel(r"D:\Srujan\Projects\Infosys\dining_info.xlsx")
        return df
    except Exception as e:
        st.error("Error loading data: " + str(e))
        return None

# Streamlit App
st.set_page_config(page_title="Multi-Cuisine Hotel", layout="wide")

st.title("Welcome to The Grand Multi-Cuisine Hotel")
st.subheader("Experience world-class dining with flavors from across the globe!")

# Sidebar for Navigation
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to", ["Home", "Cuisines", "Special Offers", "Reviews", "Contact Us"])


# Home Page
if menu == "Home":
    st.image(r"D:\Srujan\Projects\Infosys\hotel_img.jpeg", use_column_width=True)
    st.write("Enjoy luxury dining with exquisite flavors from various cultures. Our chefs bring authenticity and innovation to every dish!")
    

    # Ask if the customer has a customer_id
    has_customer_id = st.radio("Do you have a Customer ID?", ("Yes", "No"))

    if has_customer_id == "Yes":
        customer_id = st.text_input("Enter your Customer ID", "")
    else:
        # If they don't have an ID, generate one (greater than 10000)
        customer_id = random.randint(10001, 99999)
        st.write(f"Your generated Customer ID: {customer_id}")

    # User Inputs
    name = st.text_input("Enter your name", "")
    checkin_date = st.date_input("Check-in Date", min_value=date.today())
    checkout_date = st.date_input("Check-out Date", min_value=checkin_date)
    age = st.number_input("Enter your age", min_value=18, max_value=120, step=1)
    stayers = st.number_input("How many stayers in total?", min_value=1, max_value=3, step=1)
    cuisine_options = ["South Indian", "North Indian", "Multi"]
    preferred_cuisine = st.selectbox("Preferred Cuisine", cuisine_options)
    booking_options = ["Yes", "No"]
    preferred_booking = st.selectbox("Do you want to book through points?", booking_options)

    # Special Requests (Optional)
    special_requests = st.text_area("Any Special Requests? (Optional)", "")

    # Submit Button

    if st.button("Submit Booking"):
        if name and customer_id:
            # Collect data into a new row
            new_data = {
                'customer_id': customer_id,
                'Preferred Cusine': preferred_cuisine,
                'age': age,
                'check_in_date': checkin_date,
                'check_out_date': checkout_date,
                'booked_through_points':preferred_booking,
                'number_of_stayers':stayers

            }
            
            # Append the new data to the dataframe
            new_df = pd.DataFrame([new_data])


            new_df['booked_through_points'] = new_df['booked_through_points'].apply(lambda x: 1 if x=='Yes' else 0)
            
            new_df['customer_id'] = new_df['customer_id'].astype(int)

            new_df['check_in_date'] = pd.to_datetime(new_df['check_in_date'])
            new_df['check_out_date'] = pd.to_datetime(new_df['check_out_date'])
            
            db = client["hotel_guests"]
            new_bookings_collection = db["new_bookings"]
            
            # Insert the new booking data into the collection
            result = new_bookings_collection.insert_one(new_df.iloc[0].to_dict())

            
            new_df['check_in_day'] = new_df['check_in_date'].dt.dayofweek  # Monday=0, Sunday=6
            new_df['check_out_day'] = new_df['check_out_date'].dt.dayofweek
            new_df['check_in_month'] = new_df['check_in_date'].dt.month
            new_df['check_out_month'] = new_df['check_out_date'].dt.month
            new_df['stay_duration'] = (new_df['check_out_date'] - new_df['check_in_date']).dt.days
            
            



            customer_features = pd.read_excel('dining_info.xlsx')
            customer_dish = pd.read_excel('dining_info.xlsx')

            cuisine_features = pd.read_excel('dining_info.xlsx')
            cuisine_dish = pd.read_excel('dining_info.xlsx')

            data['customer_id'] = data['customer_id'].astype(int)
            customer_features['customer_id'] = customer_features['customer_id'].astype(int)
            customer_dish['customer_id'] = customer_dish['customer_id'].astype(int)


            new_df = new_df.merge(customer_features, on='customer_id', how='left')
            new_df = new_df.merge(cuisine_features, on='Preferred Cusine', how='left')
            new_df = new_df.merge(customer_dish, on='customer_id', how='left')
            new_df = new_df.merge(cuisine_dish, on='Preferred Cusine', how='left')

            new_df.drop(['customer_id','check_in_date','check_out_date'],axis=1,inplace=True)

            encoder = joblib.load('encoder.pkl')

            categorical_cols = ['Preferred Cusine', 'most_frequent_dish','cuisine_popular_dish']

            encoded_test = encoder.transform(new_df[categorical_cols])

            encoded_test_df = pd.DataFrame(
                encoded_test, columns=encoder.get_feature_names_out(categorical_cols))

            new_df = pd.concat([new_df.drop(columns=categorical_cols), encoded_test_df], axis=1)

            features = list(pd.read_excel('dining_info.xlsx')[0])

            label_encoder = joblib.load('label_encoder.pkl')

            new_df = new_df[features]

            model = joblib.load('xgb_model_dining.pkl')


            y_pred_prob = model.predict_proba(new_df)
            dish_names = label_encoder.classes_

            prob_df = pd.DataFrame(y_pred_prob, columns=dish_names)

            # Add the true dish labels (y_test) as a separate column
        
            top_3_indices = np.argsort(-y_pred_prob, axis=1)[:, :3]  # Negative for descending order

            # Retrieve dish names for the top 3 predictions
            top_3_dishes = dish_names[top_3_indices]

            # Create DataFrame with top 3 dishes
            prob_df["top_1"] = top_3_dishes[:, 0]
            prob_df["top_2"] = top_3_dishes[:, 1]
            prob_df["top_3"] = top_3_dishes[:, 2]


            # Display success message and booking information
            st.success(f"✅ Booking Confirmed for {name} (Customer ID: {customer_id})!")
            st.write(f"**Check-in:** {checkin_date}")
            st.write(f"**Check-out:** {checkout_date}")
            st.write(f"**Age:** {age}")
            st.write(f"**Preferred Cusine:** {preferred_cuisine}")
            if special_requests:
                st.write(f"**Special Requests:** {special_requests}")
            
            # Display the dataframe
            st.write("Booking Data so far:")

            dishes = [
            prob_df["top_1"].iloc[0],
            prob_df["top_2"].iloc[0],
            prob_df["top_3"].iloc[0]
            ]

            thali_dishes = [dish.lower() for dish in dishes if "thali" in dish]
            other_dishes = [dish.lower() for dish in dishes if "thali" not in dish]

            st.write("Discounts for you!")

            # Display discount messages based on the dish type
            if thali_dishes:
                st.write(f"Get 20% off on {', '.join(thali_dishes)}")
            if other_dishes:
                st.write(f"Get 15% off on {', '.join(other_dishes)}")
            
            st.write("Check your coupon code on your email!")

        else:
            st.warning("⚠️ Please enter your name and Customer ID to proceed!")



# Cuisine Section
elif menu == "Cuisines":
    st.header("Our Special Cuisines")
    cuisines = ["Italian", "Chinese", "Indian", "Mexican", "French", "Japanese"]
    selected_cuisine = st.selectbox("Choose a Cuisine", cuisines)
    st.write(f"### {selected_cuisine} Delights")
    st.write("Explore the best dishes from this cuisine prepared by expert chefs!")

# Special Offers
elif menu == "Special Offers":
    st.header("Exclusive Offers & Discounts")
    st.write("Stay tuned for seasonal discounts and special combos!")

# Customer Reviews
elif menu == "Reviews":
    st.header("What Our Guests Say")
    reviews = ["The food was amazing! - Saathvik raj", "Loved the ambiance! - Snehith Silveri", "A must-visit for food lovers - Sowmya Sri"]
    for review in reviews:
        st.write(f"📌 {review}")
    st.text_area("Leave a Review")

# Contact Section
elif menu == "Contact Us":
    st.header("Get in Touch")
    st.write("📍 Location: Hyderabad, Telangana")
    st.write("📞 Phone: +123 456 7890")
    st.write("📧 Email: info@grandhotel.com")
    
    name = st.text_input("Your Name")
    message = st.text_area("Your Message")
    if st.button("Send Message"):
        st.success("Thank you for reaching out! We'll get back to you soon.")

st.sidebar.write("---")
st.sidebar.write("© 2025 The Grand Multi-Cuisine Hotel")
