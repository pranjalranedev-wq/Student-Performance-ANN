
import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #4F46E5;
    }
    .subtitle {
        text-align: center;
        color: gray;
        font-size: 17px;
        margin-bottom: 25px;
    }
    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🎓 Student Performance Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Artificial Neural Network (ANN) Analytics Dashboard</div>',
    unsafe_allow_html=True
)

st.info(
    "Enter academic details to estimate the student's final score "
    "and explore the results through interactive visualizations."
)

# --------------------------------------------------
# LOAD MODEL AND SCALER
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("student_ann.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_resources()
except Exception as e:
    st.error(
        "Unable to load the ANN model or scaler. "
        "Check that student_ann.keras and scaler.pkl are in the "
        "same directory as app.py."
    )
    st.exception(e)
    st.stop()

# --------------------------------------------------
# SIDEBAR: STUDENT INPUTS
# --------------------------------------------------
st.sidebar.header("📝 Student Information")
st.sidebar.caption("Adjust the values to predict performance.")

study_hours = st.sidebar.slider(
    "📚 Study hours per day", 0.0, 16.0, 3.0, 0.5
)
attendance = st.sidebar.slider(
    "📅 Attendance (%)", 0, 100, 80
)
previous_marks = st.sidebar.slider(
    "📖 Previous marks", 0, 100, 70
)
assignment_marks = st.sidebar.slider(
    "✍️ Assignment marks", 0, 100, 75
)
sleep_hours = st.sidebar.slider(
    "😴 Sleep hours per day", 0.0, 12.0, 7.0, 0.5
)

predict_button = st.sidebar.button(
    "🚀 Predict Performance", use_container_width=True
)

# --------------------------------------------------
# HELPER: PERFORMANCE GAUGE
# --------------------------------------------------
def create_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            "suffix": "/100",
            "font": {"size": 34}
        },
        title={"text": "Predicted Final Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4F46E5"},
            "steps": [
                {"range": [0, 50], "color": "#FECACA"},
                {"range": [50, 75], "color": "#FDE68A"},
                {"range": [75, 100], "color": "#BBF7D0"}
            ],
            "threshold": {
                "line": {"color": "#111827", "width": 4},
                "thickness": 0.8,
                "value": 75
            }
        }
    ))

    fig.update_layout(
        height=320,
        margin=dict(l=25, r=25, t=70, b=20)
    )
    return fig

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if predict_button:

    inputs = np.array([[
        study_hours,
        attendance,
        previous_marks,
        assignment_marks,
        sleep_hours
    ]])

    try:
        scaled_inputs = scaler.transform(inputs)
        prediction = float(
            model.predict(scaled_inputs, verbose=0).reshape(-1)[0]
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    if not np.isfinite(prediction):
        st.error("The model returned an invalid prediction.")
        st.stop()

    # Assumes the model was trained to predict marks on a 0–100 scale.
    score = float(np.clip(prediction, 0, 100))

    if score >= 75:
        category = "High"
        message = "🌟 Strong predicted performance!"
        category_color = "#16A34A"
    elif score >= 50:
        category = "Medium"
        message = "📘 There may be room to improve."
        category_color = "#D97706"
    else:
        category = "Low"
        message = "💪 Focused academic support may help."
        category_color = "#DC2626"

    st.markdown("---")
    st.subheader("📊 Prediction Overview")

    # TOP METRICS
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Predicted Score", f"{score:.2f}/100")

    with col2:
        st.metric("Performance Category", category)

    with col3:
        st.metric("Attendance", f"{attendance}%")

    with col4:
        st.metric("Study Time", f"{study_hours:.1f} hrs/day")

    st.markdown(f"### {message}")

    # GAUGE AND PIE CHART
    st.markdown("---")
    st.subheader("🎯 Score Analysis")

    gauge_col, pie_col = st.columns(2)

    with gauge_col:
        st.plotly_chart(
            create_gauge(score),
            use_container_width=True
        )

    with pie_col:
        pie_fig = px.pie(
            names=["Predicted marks", "Remaining marks"],
            values=[score, 100 - score],
            hole=0.58,
            title="Predicted Score Distribution",
            color_discrete_sequence=["#4F46E5", "#E5E7EB"]
        )
        pie_fig.update_traces(
            textinfo="label+percent",
            textposition="inside"
        )
        pie_fig.update_layout(
            height=320,
            legend_title_text="Score Components",
            margin=dict(l=15, r=15, t=60, b=15)
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    # ACADEMIC INPUTS BAR CHART
    st.markdown("---")
    st.subheader("📚 Academic Profile")

    profile = pd.DataFrame({
        "Factor": [
            "Attendance",
            "Previous Marks",
            "Assignment Marks"
        ],
        "Value": [
            attendance,
            previous_marks,
            assignment_marks
        ]
    })

    profile_fig = px.bar(
        profile,
        x="Factor",
        y="Value",
        color="Factor",
        text="Value",
        title="Academic Indicators",
        range_y=[0, 110],
        color_discrete_sequence=["#6366F1", "#14B8A6", "#F59E0B"]
    )

    profile_fig.update_traces(
        textposition="outside",
        showlegend=False
    )
    profile_fig.update_layout(
        yaxis_title="Marks / Percentage",
        xaxis_title="",
        height=380,
        showlegend=False
    )

    st.plotly_chart(profile_fig, use_container_width=True)

    # INPUT SUMMARY
    st.markdown("---")
    st.subheader("📋 Student Input Summary")

    summary = pd.DataFrame({
        "Academic Factor": [
            "Study Hours per Day",
            "Attendance",
            "Previous Marks",
            "Assignment Marks",
            "Sleep Hours per Day"
        ],
        "Entered Value": [
            f"{study_hours:.1f} hours",
            f"{attendance}%",
            f"{previous_marks}/100",
            f"{assignment_marks}/100",
            f"{sleep_hours:.1f} hours"
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # PERSONALIZED SUGGESTIONS
    st.markdown("---")
    st.subheader("💡 Suggestions for Improvement")

    suggestions = []

    if study_hours < 3:
        suggestions.append(
            "📚 Consider building a consistent daily study routine."
        )

    if attendance < 75:
        suggestions.append(
            "📅 Improving attendance may help maintain learning continuity."
        )

    if previous_marks < 50:
        suggestions.append(
            "📖 Review foundational topics and practice previous questions."
        )

    if assignment_marks < 50:
        suggestions.append(
            "✍️ Spend more time practicing assignments and reviewing feedback."
        )

    if sleep_hours < 7:
        suggestions.append(
            "😴 Consider allowing more time for sleep and recovery."
        )
    elif sleep_hours > 9:
        suggestions.append(
            "😴 Review whether your sleep schedule supports your daily routine."
        )

    if score < 50:
        suggestions.append(
            "🎯 Consider a structured revision plan and support from teachers."
        )
    elif score >= 75:
        suggestions.append(
            "🏆 Maintain consistent learning habits and challenge yourself."
        )

    if not suggestions:
        suggestions.append(
            "✅ Keep monitoring your progress and maintain consistent habits."
        )

    for suggestion in suggestions:
        st.write(suggestion)

    st.caption(
        "Note: This is a model-generated estimate, not a guaranteed result. "
        "Charts describe the entered data and prediction; they do not "
        "establish that any input causes a particular score."
    )

else:
    # INITIAL DASHBOARD
    st.markdown("---")
    st.subheader("✨ Ready to Analyze Student Performance?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊")
        st.write("**Visual Analytics**")
        st.write("Explore scores using interactive charts.")

    with col2:
        st.markdown("### 🧠")
        st.write("**ANN Prediction**")
        st.write("Estimate final marks using your trained model.")

    with col3:
        st.markdown("### 💡")
        st.write("**Study Insights**")
        st.write("Review academic indicators and improvement suggestions.")

    st.info(
        "Set the student's values in the sidebar and click "
        "**Predict Performance** to generate the dashboard."
    )

    st.caption(
        "The charts and predicted score will appear after you run a prediction."
    )
