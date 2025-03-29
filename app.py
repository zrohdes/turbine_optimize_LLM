import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import google.generativeai as genai

# Configure Gemini API using Streamlit secrets
try:
    # For production in Streamlit Cloud
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # Fallback for local development
    try:
        import os
        from dotenv import load_dotenv

        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    except:
        GOOGLE_API_KEY = None
        st.error("Google API key not found. Please add it to your .streamlit/secrets.toml file or .env file.")

# Only configure if API key is available
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# Set page configuration
st.set_page_config(
    page_title="Wind Turbine Power Optimization",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Functions for data processing
def load_wind_turbine_data(uploaded_file):
    """Load and process wind turbine data from uploaded file."""
    if uploaded_file is not None:
        try:
            # Try to read as CSV first
            df = pd.read_csv(uploaded_file)
        except:
            try:
                # Try to read as Excel if CSV fails
                df = pd.read_excel(uploaded_file)
            except:
                st.error("Failed to load data. Please ensure file is CSV or Excel format.")
                return None

        # Basic data cleaning
        df = df.dropna()
        return df
    return None


def preprocess_data(df):
    """Perform preprocessing on wind turbine data."""
    if df is None:
        return None

    # Make a copy to avoid modifying the original
    processed_df = df.copy()

    # Standardize column names (lowercase and replace spaces with underscores)
    processed_df.columns = processed_df.columns.str.lower().str.replace(' ', '_')

    # Ensure required columns exist (adapt based on your data structure)
    required_cols = ['timestamp', 'wind_speed', 'wind_direction', 'power_output']
    missing_cols = [col for col in required_cols if col not in processed_df.columns]

    if missing_cols:
        # Attempt to find alternative column names
        mapping = {
            'timestamp': ['time', 'date', 'datetime'],
            'wind_speed': ['speed', 'velocity', 'wind_velocity'],
            'wind_direction': ['direction', 'angle', 'wind_angle'],
            'power_output': ['output', 'power', 'energy', 'generation']
        }

        for missing_col in missing_cols:
            alternatives = mapping.get(missing_col, [])
            for alt in alternatives:
                matching_cols = [col for col in processed_df.columns if alt in col]
                if matching_cols:
                    processed_df[missing_col] = processed_df[matching_cols[0]]
                    break

    # Convert timestamp to datetime if needed
    if 'timestamp' in processed_df.columns:
        try:
            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
        except:
            st.warning("Could not convert timestamp column to datetime.")

    return processed_df


def analyze_data(df):
    """Extract key statistics and insights from the data."""
    if df is None:
        return None

    analysis = {
        "total_records": len(df),
        "date_range": [df['timestamp'].min(), df['timestamp'].max()] if 'timestamp' in df.columns else None,
        "avg_power_output": df['power_output'].mean() if 'power_output' in df.columns else None,
        "max_power_output": df['power_output'].max() if 'power_output' in df.columns else None,
        "avg_wind_speed": df['wind_speed'].mean() if 'wind_speed' in df.columns else None,
        "correlation": df[['wind_speed', 'power_output']].corr().to_dict() if all(
            col in df.columns for col in ['wind_speed', 'power_output']) else None
    }

    return analysis


def get_gemini_optimization(df, analysis):
    """Use Gemini AI to suggest optimizations for wind turbine power generation."""
    if df is None or analysis is None:
        return "No data available for optimization analysis."

    # Prepare data summary for Gemini
    data_summary = f"""
Wind Turbine Data Summary:
- Total records: {analysis['total_records']}
- Date range: {analysis['date_range'][0]} to {analysis['date_range'][1]}
- Average power output: {analysis['avg_power_output']:.2f} kW
- Maximum power output: {analysis['max_power_output']:.2f} kW
- Average wind speed: {analysis['avg_wind_speed']:.2f} m/s
- Correlation between wind speed and power output: {analysis['correlation']['wind_speed']['power_output']:.4f}

Data sample:
{df.head(5).to_string()}
    """

    # Create prompt for Gemini
    prompt = f"""
Based on the following wind turbine data, provide specific recommendations to optimize power generation:

{data_summary}

Please provide:
1. Key insights from the data
2. Specific optimization opportunities
3. Recommended turbine parameters (pitch angle, yaw adjustment, etc.)
4. Potential power output increase estimation
5. Implementation steps for operators

Format the response as JSON with the following structure:
{{
  "insights": ["insight1", "insight2", ...],
  "optimization_opportunities": ["opportunity1", "opportunity2", ...],
  "recommended_parameters": {{"parameter1": "value1", "parameter2": "value2", ...}},
  "estimated_power_increase": "X%",
  "implementation_steps": ["step1", "step2", ...]
}}
"""

    try:
        # Generate content using Gemini Pro
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        # Parse JSON response
        try:
            recommendations = json.loads(response.text)
        except:
            # Fallback if Gemini doesn't return valid JSON
            st.warning("AI response was not in valid JSON format. Using text response instead.")
            recommendations = {"raw_response": response.text}

        return recommendations
    except Exception as e:
        return f"Error generating optimization recommendations: {str(e)}"


# Streamlit UI
def main():
    st.title("🌬️ Wind Turbine Power Optimization")
    st.write("Upload your wind turbine data to get AI-powered optimization recommendations")

    # Sidebar
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses Gemini LLM to analyze wind turbine data and "
        "provide recommendations for optimizing power generation. "
        "Upload your data to get started."
    )

    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
        1. Upload your wind turbine data (CSV or Excel)
        2. Review data statistics and visualizations
        3. Get AI-powered optimization recommendations
        4. Export results for implementation
        """
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload Wind Turbine Data", type=["csv", "xlsx", "xls"])

    if uploaded_file:
        # Load and process data
        raw_df = load_wind_turbine_data(uploaded_file)

        if raw_df is not None:
            st.success(f"Successfully loaded data with {len(raw_df)} records")

            # Process data
            df = preprocess_data(raw_df)

            # Show data preview
            st.header("Data Preview")
            st.dataframe(df.head(10))

            # Data analysis section
            st.header("Data Analysis")
            col1, col2 = st.columns(2)

            # Generate insights
            analysis = analyze_data(df)

            if analysis:
                with col1:
                    st.subheader("Key Statistics")
                    st.metric("Total Records", analysis["total_records"])
                    st.metric("Avg Power Output", f"{analysis['avg_power_output']:.2f} kW")
                    st.metric("Max Power Output", f"{analysis['max_power_output']:.2f} kW")
                    st.metric("Avg Wind Speed", f"{analysis['avg_wind_speed']:.2f} m/s")

                with col2:
                    st.subheader("Power vs Wind Speed")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(data=df, x='wind_speed', y='power_output', alpha=0.6, ax=ax)
                    sns.regplot(data=df, x='wind_speed', y='power_output', scatter=False, ax=ax)
                    st.pyplot(fig)

            # Time series analysis
            if 'timestamp' in df.columns and 'power_output' in df.columns:
                st.subheader("Power Output Over Time")
                fig, ax = plt.subplots(figsize=(12, 6))
                df.set_index('timestamp')['power_output'].plot(ax=ax)
                ax.set_xlabel('Time')
                ax.set_ylabel('Power Output (kW)')
                st.pyplot(fig)

            # AI Optimization section
            st.header("AI-Powered Optimization")

            with st.spinner("Generating optimization recommendations with Gemini AI..."):
                recommendations = get_gemini_optimization(df, analysis)

            if isinstance(recommendations, dict):
                # Display structured recommendations
                if "insights" in recommendations:
                    st.subheader("Key Insights")
                    for i, insight in enumerate(recommendations["insights"], 1):
                        st.markdown(f"**{i}.** {insight}")

                if "optimization_opportunities" in recommendations:
                    st.subheader("Optimization Opportunities")
                    for i, opportunity in enumerate(recommendations["optimization_opportunities"], 1):
                        st.markdown(f"**{i}.** {opportunity}")

                if "recommended_parameters" in recommendations:
                    st.subheader("Recommended Parameters")
                    params_df = pd.DataFrame(
                        recommendations["recommended_parameters"].items(),
                        columns=["Parameter", "Recommended Value"]
                    )
                    st.table(params_df)

                if "estimated_power_increase" in recommendations:
                    st.subheader("Potential Impact")
                    st.metric(
                        "Estimated Power Output Increase",
                        recommendations["estimated_power_increase"],
                        delta="↑"
                    )

                if "implementation_steps" in recommendations:
                    st.subheader("Implementation Steps")
                    for i, step in enumerate(recommendations["implementation_steps"], 1):
                        st.markdown(f"**Step {i}:** {step}")

                # Raw response fallback
                if "raw_response" in recommendations:
                    st.subheader("AI Recommendations")
                    st.write(recommendations["raw_response"])
            else:
                # Display text response
                st.text(recommendations)

            # Export options
            st.header("Export Results")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Export Recommendations (JSON)"):
                    if isinstance(recommendations, dict):
                        # Create JSON string
                        json_data = json.dumps(recommendations, indent=2)
                        st.download_button(
                            label="Download JSON",
                            data=json_data,
                            file_name=f"wind_turbine_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )

            with col2:
                if st.button("Export Processed Data (CSV)"):
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"processed_wind_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    else:
        # Display sample data option
        st.info("No data uploaded. You can use sample data to test the application.")
        if st.button("Load Sample Data"):
            # Generate synthetic wind turbine data
            dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='H')
            wind_speeds = np.random.normal(8, 2, len(dates))  # Mean 8 m/s, std 2 m/s
            wind_directions = np.random.uniform(0, 360, len(dates))

            # Power output based on wind speed (simplified model)
            # P = 0.5 * rho * A * Cp * v^3 (where rho=air density, A=swept area, Cp=power coefficient, v=wind speed)
            base_power = 0.5 * 1.225 * 5000 * 0.4 * (wind_speeds ** 3)
            noise = np.random.normal(0, base_power * 0.05)  # Add some noise
            power_output = base_power + noise

            # Clip power output to realistic values
            power_output = np.clip(power_output, 0, 2000)

            # Create DataFrame
            sample_df = pd.DataFrame({
                'timestamp': dates,
                'wind_speed': wind_speeds,
                'wind_direction': wind_directions,
                'power_output': power_output,
                'temperature': np.random.normal(15, 5, len(dates)),
                'humidity': np.random.uniform(40, 90, len(dates)),
                'pressure': np.random.normal(1013, 10, len(dates))
            })

            # Process and display same as uploaded data
            st.success(f"Successfully loaded sample data with {len(sample_df)} records")
            df = sample_df

            # Show data preview
            st.header("Data Preview")
            st.dataframe(df.head(10))

            # Continue with same analysis as for uploaded data
            analysis = analyze_data(df)

            if analysis:
                st.header("Data Analysis")
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Key Statistics")
                    st.metric("Total Records", analysis["total_records"])
                    st.metric("Avg Power Output", f"{analysis['avg_power_output']:.2f} kW")
                    st.metric("Max Power Output", f"{analysis['max_power_output']:.2f} kW")
                    st.metric("Avg Wind Speed", f"{analysis['avg_wind_speed']:.2f} m/s")

                with col2:
                    st.subheader("Power vs Wind Speed")
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.scatterplot(data=df, x='wind_speed', y='power_output', alpha=0.6, ax=ax)
                    sns.regplot(data=df, x='wind_speed', y='power_output', scatter=False, ax=ax)
                    st.pyplot(fig)

                # AI section for sample data
                st.header("AI-Powered Optimization")
                with st.spinner("Generating optimization recommendations with Gemini AI..."):
                    recommendations = get_gemini_optimization(df, analysis)

                # Display recommendations same as for uploaded data
                if isinstance(recommendations, dict):
                    if "insights" in recommendations:
                        st.subheader("Key Insights")
                        for i, insight in enumerate(recommendations["insights"], 1):
                            st.markdown(f"**{i}.** {insight}")

                    # Other recommendation sections follow...


# Run the app
if __name__ == "__main__":
    main()