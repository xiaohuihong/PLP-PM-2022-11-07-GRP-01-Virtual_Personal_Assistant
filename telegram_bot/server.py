from bot import telegram_chatbot
from chatterbot import ChatBot
from chatterbot.comparisons import LevenshteinDistance
from chatterbot.response_selection import get_first_response
from chatterbot.trainers import ChatterBotCorpusTrainer
from news_scrape.scrapeCNN import news_crawler
import re
import pandas as pd
from question_answering.stacked_bilstm import qa_model


bot = telegram_chatbot("config.cfg")

chatbot = ChatBot(
    'plp_newsbot',
    logic_adapters=[
        {
            "import_path": "chatterbot.logic.BestMatch",
            "statement_comparison_function": LevenshteinDistance,
            "response_selection_method": get_first_response
        }
    ]
)

trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")
news_df = pd.DataFrame()
qa_model = qa_model()
def make_reply(msg, reply_to_msg):
    replies = []
    parse_mode = 'text'
    num_news = 5
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
                replies.append("[{title}]({url})".format(title=row['title'],url=row['url']))
            parse_mode = 'MarkdownV2'
        elif msg.lower().startswith("qa:") and reply_to_msg is not None:
            print(news_df.columns)
            if "title" in news_df.columns:
                print("Starting Getting the Answer...")
                reply_df = news_df.loc[news_df["title"] == reply_to_msg]
                answer = qa_model.predict(reply_df["content"].iloc[0], msg.split(':')[-1])
                print(answer)
                replies.append("Answer: {answer}".format(answer=answer))
            else:
                replies.append(chatbot.get_response(msg))
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