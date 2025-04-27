import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import altair as alt
import numpy as np


st.set_page_config(
    page_title="Whatsapp Group Chat Analyzer",
    page_icon="🗣",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.cache_data.clear()
st.cache_resource.clear()

st.title("💭 Whatsapp Chat Analyzer")


uploaded_file = st.file_uploader("Choose a file (should be .txt format)")


if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        with st.spinner("Processing your file. Please wait..."):
            try:
                bytes_data = uploaded_file.getvalue()
                data = bytes_data.decode("utf-8")
                df, start_date, last_date = preprocessor.preprocess(data)


                user_list = df['user'].unique().tolist()
                if 'group_notification' in user_list:
                    user_list.remove('group_notification')
                user_list.sort()
                user_list.insert(0,"Everyone")

                selected_user = st.selectbox("Show analysis of:",user_list)

                if st.button("Show Analysis"):

                    num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Total Messages", f"{num_messages}")
                        st.metric("Total Words", words)
                        st.metric("Most Active Chatter", helper.get_most_active_user(df))

                    with col2:
                        st.metric("Media Shared", num_media_messages)
                        st.metric("Links Shared", num_links)
                        total_days = (last_date - start_date).days + 1
                        st.metric("Chat Duration", total_days)

                    st.markdown(f"""
                                   📈 Chat started from {start_date.strftime('%B %d, %Y')} to {last_date.strftime('%B %d, %Y')}
            
                                   🏆 **{helper.get_most_active_user(df)}** has most number of messages !
                                   
                                   """)


                    #Message Volume
                    st.write("""## Messages Volume Over Time""")

                    smoothed_daily_activity_df = helper.smoothed_daily_activity(selected_user,df=df, years=3)
                    st.area_chart(smoothed_daily_activity_df)


                    #Monthy Analysis
                    st.markdown(""" ## Monthly Analysis """)
                    timeline = helper.monthly_timeline(selected_user, df)
                    timeline.set_index("time", inplace=True)
                    st.line_chart(timeline["message"])


                    #Daily Analysis
                    st.markdown(" ## Daily Analysis")
                    daily_timeline = helper.daily_timeline(selected_user, df)
                    daily_timeline.set_index("only_date", inplace=True)
                    st.line_chart(daily_timeline["message"])


                    #Most Active Day
                    st.markdown(" ## Most active day")
                    busy_day, new_df = helper.week_activity_map(selected_user,df)
                    col1, col2 = st.columns(2)

                    with col1:
                        st.bar_chart(busy_day)

                    with col2:
                        st.dataframe(new_df)

                    #Most Active Month
                    st.markdown(" ## Most active month")
                    busy_month = helper.month_activity_map(selected_user, df)
                    st.bar_chart(busy_month)

                    if selected_user != 'Overall':
                        st.markdown("## Most Talkative Day")

                        daily_message_counts, most_talkative_day, message_count = helper.plot_most_talkative_day(df)

                        if message_count > 0:
                            st.line_chart(daily_message_counts)
                            st.markdown(
                                f"🎉 The most talkative day was **{most_talkative_day}** with **{message_count}** messages!")
                        else:
                            st.markdown("No messages found to analyze the most talkative day.")

                    #Weekly Heatmap
                    st.markdown(" ## Weekly Activity Map")
                    user_heatmap = helper.activity_heatmap(selected_user,df)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(user_heatmap, cmap="coolwarm", annot=True, fmt=".0f", linewidths=0.5, ax=ax)
                    st.pyplot(fig)

                    #Most Active Person
                    if selected_user == 'Everyone':
                        st.markdown(' ## Most Active Person')
                        x,new_df = helper.most_busy_users(selected_user,df)
                        fig, ax = plt.subplots()

                        col1, col2 = st.columns(2)

                        with col1:
                            st.bar_chart(x)
                        with col2:
                            st.dataframe(new_df)

                    # WordCloud
                    st.write("## Most Used Words (Word Cloud)")
                    st.subheader("Here are some most used words:")
                    word_cloud = helper.create_word_cloud(df)
                    st.image(word_cloud)

                    #Message Activity(day)
                    st.write("""
                                        ## Message Activity by Time of Day
                                        """)
                    time_of_day_data = helper.activity_time_of_day_ts(selected_user, df)
                    st.altair_chart(time_of_day_data)


                    #Message Activity(week)
                    st.write("""
                                        ## Message Activity by Day of Week
                                        """)

                    day_of_week_data = helper.activity_day_of_week_ts(df,selected_user)
                    st.altair_chart(day_of_week_data)

                    #Words
                    most_common_df = helper.most_common_words(selected_user, df)
                    most_common_df.columns = ["Word", "Count"]

                    chart = alt.Chart(most_common_df).mark_bar().encode(
                        x=alt.X("Count:Q", title="Frequency"),
                        y=alt.Y("Word:N", sort="-x", title="Word"),
                        color=alt.Color("Word:N", legend=None)  # Optional: different color per word
                    ).properties(
                        title="Most Common Words",
                        width=600,
                        height=400
                    )

                    st.markdown(" ## Most Common Words")
                    st.altair_chart(chart, use_container_width=True)


                    #Conversation Starter
                    st.markdown("### Conversation Starters")
                    col1, col2= st.columns(2)

                    starter_df = helper.identify_conversation_starters(selected_user, df)


                    with col1:
                        st.bar_chart(starter_df.set_index('User')['Conversation Starts'])

                    with col2:
                        st.dataframe(starter_df)



                    #Consecutive Message
                    if selected_user == 'Everyone':
                        st.write("## Consecutive Message Analysis")

                        streak_info = helper.find_longest_consecutive_streak(df, selected_user)
                        st.write(f"User with the longest streak: {streak_info['user']} 🎊")
                        st.write(f"**Streak Length:** {streak_info['streak_length']} messages")
                        st.write(f"**Start Time:** {streak_info['start_time']}")
                        st.write(f"**End Time:** {streak_info['end_time']}")

                        st.dataframe(streak_info["streak_messages"])

                        chart_data = pd.DataFrame({
                            "User": [streak_info["user"]],
                            "Streak Length": [streak_info["streak_length"]]
                        })

                        chart = alt.Chart(chart_data).mark_bar().encode(
                            x=alt.X("Streak Length:Q", title="Number of Messages"),
                            y=alt.Y("User:N", title="User"),
                            color=alt.Color("User:N", legend=None)
                        ).properties(
                            width=600,
                            height=200,
                            title="Longest Consecutive Streak"
                        )

                        st.altair_chart(chart, use_container_width=True)


                    #Response Time
                    st.write("""
                                        ## Response Time Analysis
                                        """)

                    response_time_analysis = helper.analyze_response_time(df, selected_user)

                    st.altair_chart(response_time_analysis['median_chart'], use_container_width=True)

                    slowest_responder = response_time_analysis['slowest_responder']

                    #Emoji
                    emoji_df = helper.emoji_helper(selected_user, df)
                    st.title("Emoji Analysis")
                    if not emoji_df.empty:
                        st.markdown("## Emoji Analysis")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.dataframe(emoji_df.head(20))
                        with col2:
                            fig, ax = plt.subplots()
                            ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                            st.pyplot(fig)
                    else:
                        st.markdown("No emojis found in the chat.")




                    #Sentiment Analysis
                    sentiment_counts = helper.sentiment_analysis(df, selected_user)

                    chart = alt.Chart(sentiment_counts).mark_bar().encode(
                        x=alt.X("sentiment", title="Sentiment"),
                        y=alt.Y("count", title="Message Count"),
                        color=alt.Color("sentiment", legend=None),
                    ).properties(
                        title="Sentiment Analysis",
                        width=600,
                        height=400
                    )

                    st.title("Sentiment Analysis")
                    st.altair_chart(chart, use_container_width=True)


                    #Awards
                    if selected_user == 'Everyone':
                        st.markdown("## Chat Badges 🏅")
                        badges = helper.assign_chat_badges(df)

                        for user, badge in badges.items():
                            st.markdown(f"**{user}**: {badge}")



            except Exception as e:
                st.error(f"Error during preprocessing: {e}")









