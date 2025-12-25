🌍 AI-Driven Air Quality Monitoring Dashboard
📌 Mini Project — Anurag University

Team Members:

Kalyan Nallavolu — 22EG112D46

Sri Varshith — 22EG112D44

Manas — 22EG112D60
Department: B.Tech IT — R22 Regulation

🎯 Project Objective

This project aims to build an AI-powered air quality monitoring web application that:

✔ Predicts PM2.5 based on multiple pollutant inputs using a Machine Learning model
✔ Calculates AQI (Air Quality Index) and shows severity levels
✔ Fetches live real-time station-based air quality data using the OpenAQ API
✔ Visualizes Indian air quality on an interactive map
✔ Provides EDA insights such as:

AQI bucket distribution

Most polluted & least polluted cities (current snapshot)

City-wise pollutant analysis

🧠 Machine Learning Model
Attribute	Result
ML Task	Regression
Target Variable	PM2.5
Features Used	CO, NO₂, SO₂, NH₃, O₃, PM₁₀
Algorithm Used	Pre-trained regression model (via predict.py)

📌 The predicted PM2.5 is then converted into AQI category:

Good

Satisfactory

Moderate

Poor

Very Poor

Severe

🌐 Live Data Integration

We use:

📡 OpenAQ Public API — to fetch real-time pollutant data for Indian cities
🛟 Built-in fallback dataset to ensure the dashboard always runs even without internet/API limits

➡ AQI computed internally based on pollutant levels.

📊 Exploratory Data Analysis (Interactive)

📦 AQI Bucket Chart (snapshot-based)

📈 Pearson Correlation Matrix

🏭 Most polluted cities (current API snapshot)

🌿 Least polluted cities

🏙️ City-wise pollutant bar chart

📅 Yearly AQI analysis (enabled when date/year data available)

🗺️ Map-Based Monitoring

Built using Folium

Colour-coded markers based on AQI value

Marker clustering enabled

Legend clearly indicating AQI health category

🛠️ Tech Stack
Component	Technology
Frontend	Streamlit
Model	scikit-learn
Data Source	OpenAQ API (live)
Maps	Folium + streamlit-folium
Backend Script	Python
Data Handling	Pandas
📁 Folder Structure
air_quality_project/
├── src/
│   ├── app.py                # Streamlit main UI
│   ├── predict.py            # Loads ML model + prediction logic
│   ├── stations.py           # API and fallback station data
│   ├── aqi_calculator.py     # AQI conversion logic
├── models/
│   ├── model.pkl             # ML model used for prediction
├── requirements.txt          # Dependencies
├── README.md                 # Documentation

🚀 Run Locally

Run these commands in VS Code terminal

cd air_quality_project
pip install -r requirements.txt
streamlit run src/app.py


Then open the URL:


🌍 Deployment

This application is deployable on:

Render

Streamlit Cloud

Heroku (if required)

Start command (for Render):

streamlit run src/app.py --server.port 10000 --server.address 0.0.0.0

📌 Limitations & Future Scope
Current Limitations

API provides snapshot-based AQI — not guaranteed to match CPCB official figures always

Yearly trend limited if timestamp not available in API response

Future Enhancements

🔹 Integrate proper historical database (time-series AQI)
🔹 Better AQI calculation for exact CPCB compliance
🔹 Mobile-friendly UI
🔹 Deploy as a mobile app with GPS-based AQI alerts

🏁 Conclusion

This project demonstrates how Machine Learning, APIs, and visual analytics
can be combined to make air quality information predictive, interactive, and easy to understand.

✨ Made with effort by
Team - AI Driven AQ Monitoring System
Department of IT — Anurag University
