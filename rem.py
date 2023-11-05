import streamlit as st
    from plyer import notification
    import time
    from datetime import datetime, time as dt_time

    st.markdown('<h1 style="font-size: 31px;">Medicine Reminder</h1>', unsafe_allow_html = True)

    # Create a dictionary to store medicine details (name and time)
    medicine_data = {}

    # Function to send notifications
    def send_notification(medicine_name, scheduled_time):
        current_time = datetime.now().time()
        
        if current_time >= scheduled_time:
            st.error(f"Time for {medicine_name}: {scheduled_time.strftime('%H:%M')} has already passed.")
            return

        while True:
            current_time = datetime.now().time()
            if current_time >= scheduled_time:
                st.success(f"Time to take {medicine_name}: {scheduled_time.strftime('%H:%M')}")
                notification_title = "Medicine Reminder"
                notification_message = f"It's time to take your {medicine_name}"
                notification.notify(
                    title=notification_title,
                    message=notification_message,
                    app_name="Medicine Reminder App",
                )
                break

    # Streamlit UI
    medicine_name = st.text_input("Enter the name of the medicine:")
    scheduled_time = st.time_input("Enter the time to take the medicine:")

    if st.button("Set Reminder"):
        if medicine_name and scheduled_time:
            medicine_data[medicine_name] = scheduled_time
            st.success(f"Reminder set for {medicine_name} at {scheduled_time.strftime('%H:%M')}")

    # Display current reminders
    if medicine_data:
        st.subheader("Current Reminders:")
        for name, time in medicine_data.items():
            st.write(f"{name} at {time.strftime('%H:%M')}")

    # Send reminders
    for name, time in medicine_data.items():
        send_notification(name, time)
