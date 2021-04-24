import tweepy

####input your credentials here
consumer_key = 'xxxx xxxx'
consumer_secret = 'xxxx xxxx'
access_token = 'xxxx xxxx'
access_token_secret = 'xxxx xxxx'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
# will notify user on ratelimit and will wait by it self no need of sleep.
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

date_since = "YYYY-MM-DD"

# Nairobi coordinates
geoc="-1.28333,36.81667,300km"


