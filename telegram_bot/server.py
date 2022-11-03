from bot import telegram_chatbot
from chatterbot import ChatBot
from chatterbot.comparisons import JaccardSimilarity
from chatterbot.response_selection import get_first_response
from chatterbot.trainers import ChatterBotCorpusTrainer
import re
import pandas as pd
from wordcloud import WordCloud
from news_scrape.scrapeCNN import news_crawler
from question_answering.stacked_bilstm import qa_model
from text_classification.one_hot_rnn import classification_model
from text_summarization.pretrained_t5 import summarization_model
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
wordcloud_img_path = dir_path + '/wordcloud.png'

bot = telegram_chatbot("config.cfg")
threshold = 0.9
chatbot = ChatBot(
    'plp_newsbot',
    logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": JaccardSimilarity,
            "response_selection_method": get_first_response,
            "maximum_similarity_threshold": threshold
        }
    ],
    preprocessors=[
        'chatterbot.preprocessors.clean_whitespace'
    ],
    read_only=True,
)

trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")
news_df = pd.DataFrame()
qa_model = qa_model()
classification_model = classification_model()
summarization_model = summarization_model()

def make_reply(msg, reply_to_msg):
    replies = []
    parse_mode = 'text'
    num_news = 10
    global news_df
    if msg is not None:
        if "push news" in msg.lower():
            print("Starting Crawling...")
            num = re.findall(r'\d+', msg)
            if num is not None and len(num) > 0:
                num_news = num[-1]
            crawler = news_crawler(num_news)
            news_df = crawler.get_news_df()
            for index, row in news_df.iterrows():
                if index == 0:
                    replies.append("Today's News:")
                replies.append("[【_*{cate}*_】{title}]({url})".format(
                        cate=classification_model.predict([row['title'] + " " + row['content']])[0],
                        title=row['title'],
                        url=row['url']))
            parse_mode = 'MarkdownV2'
            print("Today's News are pushed!")
        elif (msg.lower().startswith("q:") or msg.lower().startswith("question:")) \
                and reply_to_msg is not None \
                and "title" in news_df.columns:
                print("Getting the Answer...")
                reply_df = news_df.loc[news_df["title"] == reply_to_msg.split('】')[-1]]
                answer = qa_model.predict(reply_df["content"].iloc[0], msg.split(': ')[-1])
                replies.append("Answer: {answer}".format(answer=answer))
                print("The Answer is sent!")
        elif "wordcloud" in msg.lower() and "title" in news_df.columns:
            print("Generating the Wordcloud...")
            if reply_to_msg is not None:
                reply_df = news_df.loc[news_df["title"] == reply_to_msg.split('】')[-1]]
                wordcloud = WordCloud(max_font_size=40).generate(reply_df["content"].iloc[0])
                wordcloud.to_file(wordcloud_img_path)
                replies.append("The wordcloud image of this news is")
            else:
                wordcloud = WordCloud(max_font_size=40).generate(' '.join(news_df["content"]))
                wordcloud.to_file(wordcloud_img_path)
                replies.append("The wordcloud image of today's news is")
            print("The Wordcloud is sent!")
        elif ("summarize" in msg.lower() or "summarise" in msg.lower()) and "title" in news_df.columns \
                and reply_to_msg is not None:
            print("Summarizing the news...")
            reply_df = news_df.loc[news_df["title"] == reply_to_msg.split('】')[-1]]
            summary = summarization_model.predict("summarize: " + reply_df["content"].iloc[0])
            replies.append("Summary:\n{summary}".format(summary=summary))
            print("The Summary is sent!")
        else:
            replies.append(chatbot.get_response(msg))
    return replies, parse_mode


print("Now your chatbot is alive...")
update_id = None
while True:
    updates = bot.get_updates(offset=update_id)
    updates = updates["result"]
    if updates:
        for item in updates:
            update_id = item["update_id"]
            try:
                message = str(item["message"]["text"])
                reply_to_message = None
                if "reply_to_message" in item["message"]:
                    reply_to_message = str(item["message"]["reply_to_message"]["text"])
            except:
                message = None
                reply_to_message = None
            from_ = item["message"]["from"]["id"]
            replies, parse_mode = make_reply(message, reply_to_message)
            for reply in replies:
                bot.send_message(reply, from_, parse_mode)
                if reply == "The Wordcloud is":
                    bot.send_photo(from_, wordcloud_img_path)
