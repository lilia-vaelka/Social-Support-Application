import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Import our modules
from data_ingestion import DataIngestionService
from validation import ValidationService
from eligibility_model import EligibilityModel
from chatbot import SocialSupportChatbot
from models.applicant import Applicant, ApplicationStatus, DocumentType


# Page configuration
st.set_page_config(
    page_title="Social Support Application System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services
@st.cache_resource
def initialize_services():
    return {
        'data_ingestion': DataIngestionService(),
        'validation': ValidationService(),
        'eligibility_model': EligibilityModel(),
        'chatbot': SocialSupportChatbot()
    }

services = initialize_services()

# Sidebar navigation
st.sidebar.title("🏛️ Social Support System")
page = st.sidebar.selectbox(
    "Navigate",
    ["🏠 Dashboard", "📝 New Application", "📊 Analytics", "💬 Chat Assistant", "⚙️ Settings"]
)

# Main content
if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">Social Support Application Dashboard</h1>', unsafe_allow_html=True)
    
    # Load sample data
    sample_applications = services['data_ingestion'].load_mock_applications('mock_data/sample_applications.json')
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Applications", len(sample_applications))
    
    with col2:
        pending_count = sum(1 for app in sample_applications if app.status == ApplicationStatus.PENDING)
        st.metric("Pending Review", pending_count)
    
    with col3:
        # Simulate some processed applications
        approved_count = 2
        st.metric("Approved", approved_count)
    
    with col4:
        declined_count = 1
        st.metric("Declined", declined_count)
    
    # Recent applications table
    st.subheader("Recent Applications")
    
    # Create a DataFrame for display
    df_data = []
    for app in sample_applications[:5]:  # Show first 5
        df_data.append({
            "Name": app.full_name,
            "Emirates ID": app.emirates_id,
            "Income (AED)": f"{app.monthly_income:,.0f}",
            "Family Size": app.family_size,
            "Status": app.status.value.title(),
            "Application Date": app.application_date
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)
    
    # Quick stats chart
    st.subheader("Application Status Distribution")
    status_counts = {"Approved": 2, "Pending": 3, "Declined": 1}
    fig = px.pie(
        values=list(status_counts.values()),
        names=list(status_counts.keys()),
        title="Application Status Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "📝 New Application":
    st.markdown('<h1 class="main-header">New Social Support Application</h1>', unsafe_allow_html=True)
    
    # Application form
    with st.form("application_form"):
        st.subheader("Personal Information")
        
        col1, col2 = st.columns(2)
        with col1:
            full_name = st.text_input("Full Name *", placeholder="Enter your full name")
            emirates_id = st.text_input("Emirates ID *", placeholder="784-YYYY-NNNNNNN-N")
            phone_number = st.text_input("Phone Number *", placeholder="+971XXXXXXXXX")
        
        with col2:
            email = st.text_input("Email Address *", placeholder="your.email@example.com")
            date_of_birth = st.date_input("Date of Birth *", value=None, min_value=datetime(1900, 1, 1).date(),
                max_value=datetime.now().date())
            family_size = st.number_input("Family Size *", min_value=1, max_value=5, value=1)
        
        st.subheader("Financial Information")
        
        col1, col2 = st.columns(2)
        with col1:
            monthly_income = st.number_input("Monthly Income (AED) *", min_value=0.0, value=0.0)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
        
        with col2:
            monthly_expenses = st.number_input("Monthly Expenses (AED) *", min_value=0.0, value=0.0)
        
        st.subheader("Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Required Documents",
            type=['pdf', 'jpg', 'jpeg', 'png', 'csv', 'xlsx'],
            accept_multiple_files=True,
            help="Upload Emirates ID, Bank Statements, Income Certificate, and Asset List"
        )
        
        submitted = st.form_submit_button("Submit Application", type="primary")
        
        if submitted:
            if not all([full_name, emirates_id, phone_number, email, date_of_birth, monthly_income, monthly_expenses]):
                st.error("Please fill in all required fields marked with *")
            else:
                # Create applicant object
                applicant = Applicant(
                    full_name=full_name,
                    emirates_id=emirates_id,
                    phone_number=phone_number,
                    email=email,
                    date_of_birth=date_of_birth.strftime('%Y-%m-%d'),
                    monthly_income=monthly_income,
                    family_size=family_size,
                    dependents=dependents,
                    monthly_expenses=monthly_expenses,
                    application_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                # Process documents
                documents = services['data_ingestion'].process_uploaded_documents(uploaded_files) if uploaded_files else []
                
                # Validate application
                validation_result = services['validation'].validate_application(applicant, documents)
                
                # Get eligibility prediction
                eligibility_result = services['eligibility_model'].predict_eligibility(applicant, documents)
                
                # Display results
                st.success("Application submitted successfully!")
                
                # Validation results
                st.subheader("Validation Results")
                if validation_result['is_valid']:
                    st.success(f"✅ Application is valid (Score: {validation_result['score']:.1f}/100)")
                else:
                    st.error("❌ Application has validation errors")
                
                if validation_result['errors']:
                    st.error("Errors:")
                    for error in validation_result['errors']:
                        st.write(f"• {error}")
                
                if validation_result['warnings']:
                    st.warning("Warnings:")
                    for warning in validation_result['warnings']:
                        st.write(f"• {warning}")
                
                # Eligibility results
                st.subheader("Eligibility Assessment")
                
                if eligibility_result.is_eligible:
                    st.success(f"✅ **ELIGIBLE** (Confidence: {eligibility_result.confidence_score:.1%})")
                    st.success(f"Priority Level: {eligibility_result.priority_level.upper()}")
                else:
                    st.error(f"❌ **NOT ELIGIBLE** (Confidence: {eligibility_result.confidence_score:.1%})")
                
                st.write("**Reasons:**")
                for reason in eligibility_result.reasons:
                    st.write(f"• {reason}")
                
                st.write("**Recommendations:**")
                for rec in eligibility_result.recommendations:
                    st.write(f"• {rec}")
                
                # Store results in session state
                st.session_state['last_application'] = {
                    'applicant': applicant,
                    'validation': validation_result,
                    'eligibility': eligibility_result
                }

elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load sample data
    sample_applications = services['data_ingestion'].load_mock_applications('mock_data/sample_applications.json')
    
    # Income distribution
    st.subheader("Income Distribution")
    income_data = [app.monthly_income for app in sample_applications]
    fig = px.histogram(
        x=income_data,
        nbins=10,
        title="Monthly Income Distribution",
        labels={'x': 'Monthly Income (AED)', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Family size vs income
    st.subheader("Family Size vs Income")
    family_income_data = pd.DataFrame([
        {'Family Size': app.family_size, 'Monthly Income': app.monthly_income, 'Name': app.full_name}
        for app in sample_applications
    ])
    
    fig = px.scatter(
        family_income_data,
        x='Family Size',
        y='Monthly Income',
        hover_data=['Name'],
        title="Family Size vs Monthly Income",
        labels={'Family Size': 'Family Size', 'Monthly Income': 'Monthly Income (AED)'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Eligibility predictions
    st.subheader("Eligibility Predictions")
    
    # Generate predictions for sample data
    predictions = []
    for app in sample_applications:
        result = services['eligibility_model'].predict_eligibility(app, [])
        predictions.append({
            'Name': app.full_name,
            'Income per Member': app.monthly_income / app.family_size,
            'Eligible': result.is_eligible,
            'Confidence': result.confidence_score,
            'Priority': result.priority_level
        })
    
    pred_df = pd.DataFrame(predictions)
    st.dataframe(pred_df, use_container_width=True)
    
    # Feature importance
    st.subheader("Model Feature Importance")
    feature_importance = services['eligibility_model'].get_feature_importance()
    if feature_importance:
        importance_df = pd.DataFrame([
            {'Feature': feature, 'Importance': importance}
            for feature, importance in feature_importance.items()
        ]).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance in Eligibility Model"
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "💬 Chat Assistant":
    st.markdown('<h1 class="main-header">AI Chat Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize session state for chat
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_input = st.text_input("Ask me anything about social support applications:", placeholder="Type your question here...")
    
    with col2:
        if st.button("Send", type="primary"):
            if user_input:
                # Get context from last application if available
                context = {}
                if 'last_application' in st.session_state:
                    context = {
                        'application_status': 'pending',
                        'eligibility_result': st.session_state['last_application']['eligibility'].__dict__
                    }
                
                # Get chatbot response
                response = services['chatbot'].get_response(user_input, context)
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'user': user_input,
                    'bot': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Conversation")
        for message in reversed(st.session_state.chat_history[-10:]):  # Show last 10 messages
            with st.chat_message("user"):
                st.write(message['user'])
            with st.chat_message("assistant"):
                st.write(message['bot'])
            st.write(f"*{message['timestamp']}*")
            st.divider()
    
    # Quick actions
    st.subheader("Quick Actions")
    quick_actions = services['chatbot'].get_quick_actions()
    
    cols = st.columns(3)
    for i, action in enumerate(quick_actions):
        with cols[i % 3]:
            if st.button(action['label']):
                response = services['chatbot'].handle_quick_action(action['action'])
                st.session_state.chat_history.append({
                    'user': f"Clicked: {action['label']}",
                    'bot': response,
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                st.rerun()
    
    # Suggested questions
    st.subheader("Suggested Questions")
    suggested_questions = services['chatbot'].get_suggested_questions()
    
    for question in suggested_questions:
        if st.button(f"❓ {question}", key=f"suggest_{question}"):
            response = services['chatbot'].get_response(question)
            st.session_state.chat_history.append({
                'user': question,
                'bot': response,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            st.rerun()

elif page == "⚙️ Settings":
    st.markdown('<h1 class="main-header">System Settings</h1>', unsafe_allow_html=True)
    
    st.subheader("Model Configuration")
    
    # Eligibility thresholds
    st.write("**Eligibility Thresholds**")
    col1, col2 = st.columns(2)
    
    with col1:
        income_threshold = st.number_input("Income Threshold (AED per family member)", value=3000, min_value=1000, max_value=10000)
        family_size_limit = st.number_input("Maximum Family Size", value=15, min_value=5, max_value=25)
    
    with col2:
        age_min = st.number_input("Minimum Age", value=18, min_value=16, max_value=25)
        age_max = st.number_input("Maximum Age", value=65, min_value=50, max_value=80)
    
    # Model retraining
    st.subheader("Model Management")
    
    if st.button("Retrain Model", type="primary"):
        with st.spinner("Training model..."):
            # Generate new training data
            X, y = services['eligibility_model'].generate_training_data(1000)
            training_result = services['eligibility_model'].train_model(X, y)
            
            if 'error' in training_result:
                st.error(f"Training failed: {training_result['error']}")
            else:
                st.success(f"Model retrained successfully! Accuracy: {training_result['accuracy']:.2%}")
    
    # System information
    st.subheader("System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Model Status", "Trained" if services['eligibility_model'].is_trained else "Not Trained")
        st.metric("Sample Applications", len(sample_applications))
    
    with col2:
        st.metric("Validation Rules", "Active")
        st.metric("Chatbot Status", "Online")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🏛️ Government Social Support Application System | Powered by AI/ML | Prototype Version 1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
