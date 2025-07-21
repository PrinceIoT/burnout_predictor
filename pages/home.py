import streamlit as st

def load():
    st.markdown("""
        <style>
            /* Background color for the entire app */
            .stApp {
                background-color: #f0f4f8;
            }
            /* Container with white background and shadow */
            .intro-container {
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.12);
                max-width: 900px;
                margin: auto;
                margin-top: 40px;
                margin-bottom: 40px;
                text-align: center;
            }
            /* Title and subtitle styles */
            h1 {
                background-color: #007ACC;  /* Blue background */
                color: white;               /* White text */
                font-weight: 800;
                font-size: 3.5rem;
                margin-bottom: 0;
                padding: 10px 20px;
                border-radius: 10px;
                display: inline-block;
            }
            h3 {
                background-color: #007ACC;  /* Blue background */
                color: white;               /* White text */
                font-weight: 500;
                margin-top: 10px;
                margin-bottom: 30px;
                padding: 8px 16px;
                border-radius: 8px;
                display: inline-block;
            }
            /* Paragraph styles */
            p {
                font-size: 1.2rem;
                color: #4a4a4a;
                line-height: 1.6;
                max-width: 700px;
                margin-left: auto;
                margin-right: auto;
            }
            /* Image styling */
            .hero-image {
                border-radius: 12px;
                max-width: 600px;
                margin-top: 30px;
            }
            /* Responsive adjustments */
            @media (max-width: 768px) {
                h1 {
                    font-size: 2.5rem;
                }
                p {
                    font-size: 1rem;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="intro-container">
            <h1>Welcome to the Burnout Risk App</h1>
            <h3>Your partner in promoting healthy and productive workplaces</h3>
            <p>
                This application uses advanced machine learning techniques combined with behavioral data to predict burnout risk for individuals and organizations.
                Empower your team by identifying early warning signs of burnout and implementing effective interventions.
                Whether you’re an employee seeking personal insight or an HR professional analyzing workforce wellbeing, this tool is designed for you.
            </p>
            <img class="hero-image" src="https://images.unsplash.com/photo-1522071820081-009f0129c71c?auto=format&fit=crop&w=800&q=80" alt="Work Life Balance">
        </div>
        """,
        unsafe_allow_html=True,
    )

load()
