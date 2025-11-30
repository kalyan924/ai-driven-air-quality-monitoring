# src/app.py

import folium
import streamlit as st
import pandas as pd
import requests  # kept for future use if needed

from predict import load_model, make_prediction
from aqi_calculator import calculate_aqi
from stations import get_station_data
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# -------------------------------
# 🌍 MAP CREATION
# -------------------------------
def create_aqi_map(df: pd.DataFrame):
    """Create an interactive AQI map over India using Folium."""
    m = folium.Map(
        location=[22.9734, 78.6569],  # Center of India
        zoom_start=5,
        tiles="CartoDB positron",
    )

    marker_cluster = MarkerCluster().add_to(m)

    def get_color(aqi):
        if aqi <= 50:
            return "green"
        elif aqi <= 100:
            return "yellow"
        elif aqi <= 200:
            return "orange"
        elif aqi <= 300:
            return "red"
        elif aqi <= 400:
            return "purple"
        else:
            return "maroon"

    for _, row in df.iterrows():
        popup = folium.Popup(
            html=f"""
            <div style='font-size: 14px;'>
                <b>City:</b> {row['City']}<br>
                <b>PM2.5:</b> {row['PM2.5']} µg/m³<br>
                <b>PM10:</b> {row['PM10']} µg/m³<br>
                <b>AQI:</b> {row['AQI']}
            </div>
            """,
            max_width=250,
        )

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=8,
            popup=popup,
            color=get_color(row["AQI"]),
            fill=True,
            fill_color=get_color(row["AQI"]),
            fill_opacity=0.8,
        ).add_to(marker_cluster)

    # AQI Legend
    legend_html = """
     <div style="
         position: fixed;
         top: 100px;
         right: 20px;
         width: 190px;
         height: auto;
         border: 2px solid #ccc;
         z-index: 9999;
         font-size: 12px;
         background: black;
         color: white;
         padding: 10px;
         line-height: 1.4;
         border-radius: 8px;
     ">
         <b>AQI Legend</b><br>
         <i style="background:green; width:10px; height:10px; display:inline-block;"></i> Good (0–50)<br>
         <i style="background:yellow; width:10px; height:10px; display:inline-block;"></i> Satisfactory (51–100)<br>
         <i style="background:orange; width:10px; height:10px; display:inline-block;"></i> Moderate (101–200)<br>
         <i style="background:red; width:10px; height:10px; display:inline-block;"></i> Poor (201–300)<br>
         <i style="background:purple; width:10px; height:10px; display:inline-block;"></i> Very Poor (301–400)<br>
         <i style="background:maroon; width:10px; height:10px; display:inline-block;"></i> Severe (400+)
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# -------------------------------
# 🧠 MAIN APP
# -------------------------------
def main():
    st.set_page_config(page_title="AI-Driven Air Quality Monitoring", layout="wide")

    # ---------- Global CSS ----------
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1200px !important;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .info-card {
            padding: 16px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            background: linear-gradient(135deg, #111827, #020617);
            box-shadow: 0 18px 35px rgba(0,0,0,0.45);
        }
        .info-card h3 {
            margin: 0 0 6px 0;
            font-size: 1.1rem;
        }
        .info-card p {
            margin: 0;
            font-size: 0.9rem;
            color: #e5e7eb;
        }
        .footer-text {
            font-size: 13px;
            color: #9ca3af;
            text-align: center;
            padding-top: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("AI-Driven AQ Monitoring")

    st.sidebar.markdown(
        """
**Project Title**  
_AI-Driven Air Quality Monitoring_  

**Team Members**  
- Kalyan Nallavolu (22EG112D46)  
- Sri Varshith (22EG112D44)  
- Manas (22EG112D60)  

**College**  
Anurag University
        """
    )

    page = st.sidebar.radio(
        "Navigation",
        ("Overview & Prediction", "Live Data, EDA & Map", "About Project"),
    )

    # Common header on every page
    st.markdown(
        """
        <div style="padding: 10px 0 5px 0;">
            <h1 style="margin-bottom: 0;">🌍 AI-Driven Air Quality Monitoring</h1>
            <p style="color: #bbbbbb; margin-top: 4px;">
                A mini-project that combines machine learning, real-time environmental APIs, and interactive visual analytics
                for smarter air quality decision-making.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # PAGE 1: Overview & Prediction
    # =========================================================
    if page == "Overview & Prediction":
        st.subheader("🧪 ML-based PM2.5 & AQI Prediction")

        # Feature highlight cards
        col_feat1, col_feat2, col_feat3 = st.columns(3)
        with col_feat1:
            st.markdown(
                """
                <div class="info-card">
                    <h3>🤖 Machine Learning</h3>
                    <p>Predicts PM₂.₅ concentration from multiple pollutants using a trained regression model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_feat2:
            st.markdown(
                """
                <div class="info-card">
                    <h3>📊 AQI Categorisation</h3>
                    <p>Converts PM values into AQI category (Good, Moderate, Poor, etc.) for easy interpretation.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_feat3:
            st.markdown(
                """
                <div class="info-card">
                    <h3>🎯 What-if Analysis</h3>
                    <p>Change pollutant levels and instantly see how air quality prediction responds.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")

        col1, col2, col3 = st.columns(3)
        with col1:
            co = st.number_input("CO (mg/m³)", value=0.5)
            nh3 = st.number_input("NH₃ (µg/m³)", value=10.0)
        with col2:
            no2 = st.number_input("NO₂ (µg/m³)", value=30.0)
            ozone = st.number_input("O₃ (µg/m³)", value=40.0)
        with col3:
            pm10 = st.number_input("PM₁₀ (µg/m³)", value=80.0)
            so2 = st.number_input("SO₂ (µg/m³)", value=5.0)

        input_data = {
            "CO": co,
            "NH3": nh3,
            "NO2": no2,
            "OZONE": ozone,
            "PM10": pm10,
            "SO2": so2,
        }

        model, features = load_model()

        if st.button("🌟 Predict Now"):
            predicted_pm25 = make_prediction(model, features, input_data)

            concentrations = {"PM2.5": predicted_pm25, "PM10": pm10}
            aqi = calculate_aqi(concentrations)

            if aqi <= 50:
                category = "Good"
                color = "#009865"
            elif aqi <= 100:
                category = "Satisfactory"
                color = "#A3C853"
            elif aqi <= 200:
                category = "Moderate"
                color = "#FFD834"
            elif aqi <= 300:
                category = "Poor"
                color = "#FF9834"
            elif aqi <= 400:
                category = "Very Poor"
                color = "#D64E33"
            else:
                category = "Severe"
                color = "#7E0023"

            st.markdown(
                f"""
                <div style="padding:24px; background-color:{color}; border-radius:12px; text-align:center;">
                    <h2 style="color:white; margin-bottom: 4px;">Predicted PM2.5: {predicted_pm25:.2f} µg/m³</h2>
                    <h3 style="color:white; margin-top: 0;">AQI: {aqi} — {category}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            """
**How to explain this page in viva:**  
- “This is the AI / ML part of the project.”  
- “We take six input pollutants and use a pretrained regression model to estimate PM₂.₅.”  
- “Then we convert PM₂.₅ and PM₁₀ into an AQI value and category, which is what a normal user understands.”
            """
        )

    # =========================================================
    # PAGE 2: Live Data, EDA & Map
    # =========================================================
    elif page == "Live Data, EDA & Map":
        # First, get station data once (from OpenAQ API or fallback)
        df = get_station_data()

        # ---------- Live City AQI (fixed with session_state) ----------
        st.subheader("🌐 Live City AQI from Public API Snapshot (OpenAQ)")

        # Initialize session state for search
        if "searched_city" not in st.session_state:
            st.session_state.searched_city = None
        if "searched_aqi" not in st.session_state:
            st.session_state.searched_aqi = None
        if "search_error" not in st.session_state:
            st.session_state.search_error = None

        city_input = st.text_input("Enter City Name", "Mumbai")

        if st.button("Fetch City AQI"):
            st.session_state.searched_city = city_input.title()

            if "City" in df.columns and "AQI" in df.columns:
                # Case-insensitive match, allow partial
                mask = df["City"].str.lower().str.contains(city_input.strip().lower())
                city_subset = df[mask]

                if not city_subset.empty:
                    st.session_state.searched_aqi = city_subset["AQI"].mean()
                    st.session_state.search_error = None
                else:
                    st.session_state.searched_aqi = None
                    st.session_state.search_error = "❌ City not found in current OpenAQ snapshot."
            else:
                st.session_state.searched_aqi = None
                st.session_state.search_error = "❌ City/AQI columns missing in dataset."

        # Always display last searched result
        if st.session_state.search_error:
            st.error(st.session_state.search_error)
        elif st.session_state.searched_aqi is not None:
            st.markdown(
                f"""
                <div style="padding:20px; background-color:#2563EB; border-radius:10px; text-align:center;">
                    <h3 style="color:white; margin:0;">
                        Approx. AQI in {st.session_state.searched_city}: {st.session_state.searched_aqi:.0f}
                    </h3>
                    <p style="color:#e5e7eb; margin-top:4px; font-size:13px;">
                        (Based on latest station data fetched via OpenAQ API)
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ---------- Stations + Insights ----------
        st.subheader("📍 Monitoring Stations Overview (Live / Fallback)")
        st.dataframe(df, use_container_width=True)

        # Insight cards from df
        if "AQI" in df.columns and "City" in df.columns:
            try:
                max_idx = df["AQI"].idxmax()
                min_idx = df["AQI"].idxmin()
                most_city = df.loc[max_idx, "City"]
                most_aqi = df.loc[max_idx, "AQI"]
                least_city = df.loc[min_idx, "City"]
                least_aqi = df.loc[min_idx, "AQI"]

                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <h3>🏭 Most Polluted (Current Snapshot)</h3>
                            <p><b>{most_city}</b> with AQI ≈ <b>{most_aqi}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with col_i2:
                    st.markdown(
                        f"""
                        <div class="info-card">
                            <h3>🌿 Cleanest (Current Snapshot)</h3>
                            <p><b>{least_city}</b> with AQI ≈ <b>{least_aqi}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

        st.markdown("### 📊 Exploratory Data Analysis")

        # Optional BTX feature
        btx_cols = ["Benzene", "Toluene", "Xylene"]
        if all(col in df.columns for col in btx_cols):
            df["BTX"] = df["Benzene"] + df["Toluene"] + df["Xylene"]

        analysis_option = st.selectbox(
            "Choose analysis type:",
            (
                "AQI bucket chart",
                "Pearson's correlations",
                "Most polluted cities",
                "Least polluted cities",
                "City-wise pollutants analysis",
                "Yearly AQI trend",
            ),
        )

        # 2. AQI bucket chart
        if analysis_option == "AQI bucket chart":
            st.subheader("📦 AQI Bucket Chart")
            if "AQI" in df.columns:
                bins = [0, 50, 100, 200, 300, 400, 500]
                labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
                df["AQI_Bucket"] = pd.cut(
                    df["AQI"], bins=bins, labels=labels, include_lowest=True
                )
                st.bar_chart(df["AQI_Bucket"].value_counts().sort_index())
            else:
                st.warning("AQI column not found in data.")

        # 3. Pearson correlations
        elif analysis_option == "Pearson's correlations":
            st.subheader("📈 Pearson's Correlation Matrix")
            pollutants = [
                col
                for col in [
                    "PM2.5",
                    "PM10",
                    "NO2",
                    "SO2",
                    "CO",
                    "NH3",
                    "OZONE",
                    "AQI",
                    "BTX",
                ]
                if col in df.columns
            ]
            if len(pollutants) >= 2:
                st.dataframe(
                    df[pollutants].corr().style.background_gradient(cmap="coolwarm")
                )
            else:
                st.warning("Not enough numeric pollutant columns for correlation analysis.")

        # 4. Most polluted cities
        elif analysis_option == "Most polluted cities":
            st.subheader("🏭 Most Polluted Cities (by Average AQI)")
            if "City" in df.columns and "AQI" in df.columns:
                city_aqi = (
                    df.groupby("City")["AQI"]
                    .mean()
                    .sort_values(ascending=False)
                    .head(10)
                )
                st.bar_chart(city_aqi)
                st.dataframe(city_aqi.rename("Average AQI"))
            else:
                st.warning("City or AQI column missing for this analysis.")

        # 5. Least polluted cities
        elif analysis_option == "Least polluted cities":
            st.subheader("🌿 Least Polluted Cities (by Average AQI)")
            if "City" in df.columns and "AQI" in df.columns:
                city_aqi = (
                    df.groupby("City")["AQI"]
                    .mean()
                    .sort_values(ascending=True)
                    .head(10)
                )
                st.bar_chart(city_aqi)
                st.dataframe(city_aqi.rename("Average AQI"))
            else:
                st.warning("City or AQI column missing for this analysis.")

        # 6. City-wise pollutants analysis
        elif analysis_option == "City-wise pollutants analysis":
            st.subheader("🏙️ City-wise Pollutants")
            if "City" in df.columns:
                city_selected = st.selectbox("Select City", sorted(df["City"].unique()))
                city_df = df[df["City"] == city_selected]
                mean_vals = city_df.mean(numeric_only=True)
                if not mean_vals.empty:
                    st.bar_chart(mean_vals)
                    st.dataframe(mean_vals.rename("Average Value"))
                else:
                    st.warning("No numeric data available for this city.")
            else:
                st.warning("City column missing in dataset.")

        # 7. Yearly AQI trend
        elif analysis_option == "Yearly AQI trend":
            st.subheader("📅 Yearly AQI Trend")
            df_year = df.copy()
            if "Year" in df_year.columns:
                pass
            elif "Date" in df_year.columns:
                df_year["Year"] = pd.to_datetime(df_year["Date"]).dt.year
            else:
                df_year["Year"] = None

            if "Year" in df_year.columns and df_year["Year"].notnull().any():
                if "AQI" in df_year.columns:
                    yearly_aqi = df_year.groupby("Year")["AQI"].mean().sort_index()
                    st.line_chart(yearly_aqi)
                    st.dataframe(yearly_aqi.rename("Average AQI"))
                else:
                    st.warning("AQI column missing for yearly analysis.")
            else:
                st.warning(
                    "No 'Year' or 'Date' column in the dataset, so yearly trend cannot be computed."
                )

        st.markdown("---")
        st.subheader("🗺️ Air Quality Monitoring Stations Map")

        aqi_map = create_aqi_map(df)
        st_folium(aqi_map, width=1200, height=600)

        st.markdown(
            """
            <div style='font-size: 14px; color: gray; padding-top: 10px;'>
            ⚠️ This is a model-based + API-assisted dashboard and may have errors.
            For official data, please refer to the
            <a href='https://airquality.cpcb.gov.in/AQI_India' target='_blank' style='color: lightblue;'>
            CPCB AQI portal</a>.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================
    # PAGE 3: About Project
    # =========================================================
    elif page == "About Project":
        st.subheader("📃 Project Overview")

        st.markdown(
            """
**Title:** AI-Driven Air Quality Monitoring  

**Problem Statement:**  
Rapid urbanisation and industrialisation are degrading air quality in Indian cities.  
Citizens and policymakers need **real-time, interpretable and predictive tools** instead of just static AQI numbers.  

**Objectives:**  
- Use **machine learning** to predict PM₂.₅ from multiple pollutants.  
- Convert predictions into **AQI categories** that are easy to understand.  
- Integrate **live air quality APIs** (OpenAQ snapshot) to provide up-to-date information.  
- Offer **interactive analysis**: most polluted cities, cleanest cities, and city-wise pollutant profiles.
            """
        )

        st.markdown("### 🔧 Technologies Used")
        st.markdown(
            """
- **Language**: Python  
- **Framework**: Streamlit  
- **Machine Learning**: scikit-learn (model loaded via `predict.py`)  
- **Mapping**: Folium + Streamlit-Folium  
- **APIs**:  
  - OpenAQ API for live station-wise pollutants (implemented in `stations.py`)  
- **Other**: Pandas, Requests, Random (for fallback data)
            """
        )

        st.markdown("### 🔁 System Workflow")
        st.markdown(
            """
1. **Data Collection Layer**  
   - Fetches live data from OpenAQ API (country = IN).  
   - If API fails, falls back to a curated list of major Indian cities with synthetic but realistic values.

2. **AI / ML Layer**  
   - Inputs: CO, NO₂, SO₂, NH₃, O₃, PM₁₀.  
   - Model predicts PM₂.₅.  
   - AQI is then calculated using a utility function based on PM values.

3. **Visualization & Analytics Layer**  
   - Station-wise table.  
   - AQI bucket distribution (Good, Moderate, Poor…).  
   - Most / least polluted cities.  
   - City-wise pollutant averages.  
   - Interactive Folium map with AQI-coloured markers.

4. **User Interaction Layer**  
   - Simple Streamlit web UI.  
   - What-if sliders for ML prediction.  
   - City-wise AQI lookup from current API snapshot.
            """
        )

        st.markdown("### 🚀 Future Enhancements")
        st.markdown(
            """
- Expose model training pipeline and show **performance metrics** (MAE, RMSE, R²).  
- Store historical AQI snapshots to enable **true monthly / yearly trend graphs**.  
- Integrate alerts (email / SMS) when AQI exceeds a threshold.  
- Deploy the application publicly on a cloud platform.
            """
        )

        st.markdown("---")
        st.markdown(
            """
<div class="footer-text">
Mini Project • Department of CSE/IT • Anurag University  
</div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
