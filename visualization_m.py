import pandas as pd
import matplotlib.pyplot as plt

# TODO: Load dataset; replace this csv to your file
df = pd.read_csv("US_Accidents_Cleaned.csv")

# Step 1: Handling the format
# TODO: Remove extra precision if exists

# TODO: Extract relevant time-based features

# TODO: Fix missing values for numerical columns
# TODO: Ensure Severity is numeric



# Pie Charts
severity_counts = df['Severity'].value_counts(normalize=True) * 10
print(severity_counts)

sizes = [0.117344, 8.523721, 1.114469, 0.244466]
labels = ["1", "2", "3", "4"]

plt.figure(figsize=(6,6))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
plt.title("Severity Distribution Pie Chart")
plt.show()



# # Road conditions presence
# road_conditions = ['Crossing', 'Traffic_Signal', 'Junction']
# for condition in road_conditions:
    

# # Bar Plots
# # Accident Cases vs Hours
# hourly_counts = df['Hour'].value_counts().sort_index()


# # Accident Cases vs Months


# # Accident Cases vs Different Temperature


# # Accident Cases vs Different Humidity


# # Accident Cases vs Wind Speed
