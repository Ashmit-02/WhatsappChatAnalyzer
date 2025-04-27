import alt
from nltk.corpus import stopwords
from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.ndimage import gaussian_filter
import altair as alt
import numpy as np
import nltk
nltk.download('stopwords')


extract = URLExtract()


def fetch_stats(selected_user, df):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        words.extend(message.split())

    num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

    links = []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages, len(words), num_media_messages, len(links)


def most_busy_users(selected_user, df):
    df = df[df['user'] != 'group_notification']

    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={'index': 'name', 'user': 'percent'})
    return x, df


def create_word_cloud(df: pd.DataFrame):
    with open('stop_hinglish.txt', 'r') as f:
        stop_words = set(f.read().splitlines())
        
    df = df[df['user']!= 'null']
    df = df[df['user']!= 'Null']
    df = df[~df['message'].str.contains('<Media omitted>', na=False)]
    df = df[df['user'] != 'group_notification']

    def remove_emojis(text):
        return emoji.replace_emoji(text, replace='')
    def remove_stop_words(message):
        return ' '.join(word for word in message.split() if word.lower() not in stop_words)

    df['message'] = df['message'].apply(remove_stop_words)
    df['message'] = df['message'].apply(remove_emojis)

    all_words = ' '.join(df['message'].astype(str)).split()

    word_freq = pd.Series(all_words).value_counts().reset_index()
    word_freq.columns = ['word', 'frequency']

    word_freq = word_freq.head(100)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
        dict(zip(word_freq['word'], word_freq['frequency'])))

    img = wordcloud.to_image()

    return img





def most_common_words(selected_user, df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    words = []

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)

    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df


def identify_conversation_starters(selected_user,df, gap_minutes=30):
    df = df[df['user'] != 'group_notification']

    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    df['date'] = pd.to_datetime(df['date'])

    df = df.sort_values(by='date')

    df['time_diff'] = df['date'].diff().dt.total_seconds() / 60

    df['conversation_start'] = df['time_diff'] > gap_minutes

    conversation_starters = df[df['conversation_start']]['user']

    starter_counts = conversation_starters.value_counts().reset_index()
    starter_counts.columns = ['User', 'Conversation Starts']

    return starter_counts


def emoji_helper(selected_user, df):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df


def monthly_timeline(selected_user, df):
    if selected_user != "Everyone":
        df = df[df["user"] == selected_user]

    timeline = df.groupby(["year", "month_num", "month"]).count()["message"].reset_index()

    timeline["year"] = timeline["year"].astype(int)
    timeline["month_num"] = timeline["month_num"].astype(int)
    timeline["time"] = pd.to_datetime(timeline["year"].astype(str) + "-" + timeline["month_num"].astype(str), format="%Y-%m")

    timeline = timeline.sort_values("time")

    return timeline

def daily_timeline(selected_user, df):
    if selected_user != "Everyone":
        df = df[df["user"] == selected_user]

    messages_daily = df.groupby("only_date").count()["message"].reset_index()

    messages_daily["only_date"] = pd.to_datetime(messages_daily["only_date"])

    messages_daily = messages_daily.sort_values("only_date")

    return messages_daily


def week_activity_map(selected_user, df):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    x = df["day_name"].value_counts()

    df = round((df["day_name"].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
        columns={"index": "weekday", "day_name": "day_name"}
    )

    return x, df




def month_activity_map(selected_user, df):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()


def activity_heatmap(selected_user, df):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index="day_name", columns="period", values="message", aggfunc="count").fillna(0)

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    user_heatmap = user_heatmap.reindex(day_order)

    return user_heatmap


def calculate_response_times(selected_user, df):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    df['response_time'] = df['date'].diff().dt.total_seconds() / 60  # Convert difference to minutes
    df['response_time'] = df['response_time'].where(df['response_time'] < 1440)
    return df


def calculate_silent_periods(selected_user, df):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]
    df['gap_hours'] = df['date'].diff().dt.total_seconds() / 3600  # Convert gap to hours
    silent_periods = df['gap_hours'][df['gap_hours'] > 1]  # Filter gaps > 1 hour
    return silent_periods


def get_most_active_user(df: pd.DataFrame):

    if df.empty or 'user' not in df.columns or 'message' not in df.columns:
        return "No active user"

    message_counts = df.groupby("user")["message"].count().reset_index()

    if message_counts.empty:
        return "No active user"

    most_active = message_counts.sort_values("message", ascending=False).iloc[0]

    return most_active["user"]


def smoothed_daily_activity( selected_user, df: pd.DataFrame, years: int = 3):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    df["year"] = df["date"].dt.year
    min_year = df.year.max() - years
    daily_activity_df = df.loc[df["year"] > min_year].groupby(
        ['user',
         'date']).first().unstack(
        level=0).resample('D').sum(numeric_only=True).msg_length.fillna(0)

    smoothed_daily_activity_df = pd.DataFrame(
        gaussian_filter(daily_activity_df,
                        (6, 0)),
        index=daily_activity_df.index,
        columns=daily_activity_df.columns)
    return smoothed_daily_activity_df


def activity_time_of_day_ts(selected_user, df: pd.DataFrame):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    if df.empty:
        return None

    a = df.groupby([df.date.dt.hour, df.date.dt.minute, 'user'])['msg_length'].sum().unstack(fill_value=0)

    a = a.reindex(pd.MultiIndex.from_product([range(24), range(60)], names=['hour', 'minute']), fill_value=0)

    a = a.interpolate(method="linear", axis=0)
    a = pd.concat([a.tail(120), a, a.head(120)])

    smoothed = pd.DataFrame(gaussian_filter(a.values, (3, 0)), index=a.index, columns=a.columns)

    smoothed = smoothed.iloc[120:-120]

    smoothed = smoothed.reset_index()
    smoothed['time'] = pd.to_datetime(smoothed['hour'].astype(str) + ':' + smoothed['minute'].astype(str).str.zfill(2))

    melted = smoothed.melt(id_vars=['hour', 'minute', 'time'], var_name='user', value_name='activity')

    chart = alt.Chart(melted).mark_line(interpolate='monotone').encode(
        x=alt.X('time:T', title='Time of Day', axis=alt.Axis(format='%H:%M')),
        y=alt.Y('activity:Q', title='Activity'),
        color=alt.Color('user:N', title='User')
    ).properties(
        width=800,
        height=400
    )

    return chart


def activity_day_of_week_ts(df: pd.DataFrame, selected_user):
    df = df[df['user'] != 'group_notification']

    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    if df.empty:
        return None

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    o = df.groupby([df.date.dt.dayofweek, df.user])['msg_length'].sum().unstack(fill_value=0)
    o.index = pd.CategoricalIndex(o.index.map(lambda x: days[int(x)]), categories=days, ordered=True)
    o = o.sort_index()

    o_normalized = o.div(o.sum(axis=0), axis=1)

    o_normalized.index.name = 'day_of_week'

    o_melted = o_normalized.reset_index().melt(id_vars='day_of_week', var_name='user', value_name='activity')

    chart = alt.Chart(o_melted).mark_rect().encode(
        x=alt.X('day_of_week:N', sort=days, title='Day of Week'),
        y=alt.Y('user:N', title='Author'),
        color=alt.Color('activity:Q', scale=alt.Scale(scheme='viridis'), title='Activity'),
        tooltip=[
            alt.Tooltip('day_of_week:N', title='Day'),
            alt.Tooltip('user:N', title='Author'),
            alt.Tooltip('activity:Q', title='Activity', format='.1%')
        ]
    ).properties(
        width=600,
        height=400
    )

    text = chart.mark_text(baseline='middle').encode(
        text=alt.Text('activity:Q', format='.1%'),
        color=alt.condition(
            alt.datum.activity > 0.15,
            alt.value('white'),
            alt.value('black')
        )
    )

    return chart + text


def sentiment_analysis(df, selected_user):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    if df.empty:
        return None

    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["message"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sentiment"] = df["sentiment_score"].apply(lambda x: "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral")

    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    return sentiment_counts


def find_longest_consecutive_streak(df: pd.DataFrame, selected_user):
    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    if df.empty:
        return None

    df = df.sort_values('date')

    df['author_change'] = (df['user'] != df['user'].shift()).cumsum()

    grouped = df.groupby(['user', 'author_change'])

    streak_info = grouped.size().reset_index(name='streak_length')
    longest_streak = streak_info.loc[streak_info['streak_length'].idxmax()]

    max_spammer = longest_streak['user']
    max_spam = longest_streak['streak_length']

    streak_data = grouped.get_group((max_spammer, longest_streak['author_change']))
    start_time = streak_data['date'].min()
    end_time = streak_data['date'].max()

    streak_messages = streak_data[['date', 'user', 'message']]

    return {
        'user': max_spammer,
        'streak_length': max_spam,
        'start_time': start_time,
        'end_time': end_time,
        'streak_messages': streak_messages
    }

def analyze_response_time(df: pd.DataFrame, selected_user):
    df = df[df['user'] != 'group_notification']

    if selected_user != 'Everyone':
        df = df[df['user'] == selected_user]

    df = df.sort_values(["date", "user"])
    df['time_diff'] = df['date'].diff().dt.total_seconds()
    df['same_author'] = df['user'] == df['user'].shift()

    response_data = df[~((df['time_diff'] < 180) & df['same_author'])]

    response_data['response_time'] = response_data['time_diff'] / 60

    response_data['log_response_time'] = np.log10(response_data['response_time'])

    median_response_time = response_data.groupby('user')['response_time'].median().reset_index()


    median_chart = alt.Chart(median_response_time).mark_bar().encode(
        y=alt.Y('user:N', sort='-x'),
        x=alt.X('response_time:Q', title='Median Response Time (minutes)'),
        color='user:N'
    )

    return {
        'median_chart': median_chart,
        'slowest_responder': median_response_time.loc[median_response_time['response_time'].idxmax(), 'user']
    }

def plot_most_talkative_day(df):
    df['date'] = pd.to_datetime(df['date'])

    daily_message_counts = df.groupby(df['date'].dt.date)['message'].count()

    most_talkative_day = daily_message_counts.idxmax()
    highest_message_count = daily_message_counts.max()

    return daily_message_counts, most_talkative_day, highest_message_count


def assign_chat_badges(df):
    badges = {}

    df = df[df['user'] != 'group_notification']

    def count_emojis(message):
        return sum(1 for char in message if char in emoji.EMOJI_DATA)
    df['emoji_count'] = df['message'].apply(count_emojis)
    emoji_master = df.groupby('user')['emoji_count'].sum().idxmax()
    badges[emoji_master] = "Emoji Master 🏆"

    gif_messages = df[df['message'] == '<Media omitted>\n']
    gif_guru = gif_messages['user'].value_counts().idxmax() if not gif_messages.empty else None
    if gif_guru:
        badges[gif_guru] = "GIF Guru 🎥"

    df['hour'] = pd.to_datetime(df['date']).dt.hour
    late_night_messages = df[(df['hour'] >= 0) & (df['hour'] < 6)]
    late_night_texter = late_night_messages['user'].value_counts().idxmax() if not late_night_messages.empty else None
    if late_night_texter:
        badges[late_night_texter] = "Late Night Texter 🌙"

    top_chatter = df['user'].value_counts().idxmax()
    badges[top_chatter] = "Top Chatter 💬"

    silent_observer = df['user'].value_counts().idxmin()
    badges[silent_observer] = "Silent Observer 🤐"

    return badges
