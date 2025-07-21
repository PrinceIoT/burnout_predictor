import streamlit as st

def load():
    st.markdown("""
        <style>
            /* h1 and h2 white */
            h1, h2 {
                color: white !important;
                text-align: center;
            }
            /* h3 black */
            h3 {
                color: black !important;
                text-align: center;
            }
            /* Paragraphs with class tip black */
            p.tip {
                color: black !important;
                font-size: 17px;
                line-height: 1.6;
            }
            .habit-section {
                padding: 20px;
                background-color: #f0f4f8;
                border-radius: 10px;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧘 Healthy Work Habits")
    st.image("assets/work-life.jpeg", use_container_width=True)

    st.markdown("## Tips for Preventing Burnout")
    
    with st.container():
        st.markdown("""
        <div class='habit-section'>
            <h3>1. Take Regular Breaks</h3>
            <p class='tip'>Use the Pomodoro Technique: 25 minutes focused work, followed by a 5-minute break. Take longer breaks every 2 hours to recharge mentally.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='habit-section'>
            <h3>2. Set Boundaries on After-Hours Work</h3>
            <p class='tip'>Avoid checking emails or Slack after hours. Turn off notifications and communicate your availability clearly.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='habit-section'>
            <h3>3. Prioritize Sleep and Exercise</h3>
            <p class='tip'>Sleep 7–9 hours daily and get regular physical activity. Physical health greatly affects mental performance and resilience.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='habit-section'>
            <h3>4. Practice Mindfulness</h3>
            <p class='tip'>Meditation, journaling, and gratitude practices can reduce stress and improve focus. Start with 5 minutes per day.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='habit-section'>
            <h3>5. Ask for Help</h3>
            <p class='tip'>Don't hesitate to talk to your manager, HR, or a mental health professional when feeling overwhelmed. You're not alone.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.success("💡 Tip: Healthy habits practiced consistently can lower burnout risk by up to 40%!")

load()
