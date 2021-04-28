# HateSpeechDetection

## _Description_ ##

In an increasingly digital era where online social interactions are considered part of the social context, it is proving inevitable that machine learning should be used to protect people from harmful content. This has been evidenced by the multitude of instances where hate speech propagated online has led to physical injury and loss of lives across the world. Government institutions should now consider online interactions as spaces where potential crimes may occur just like in the physical world.
The goal of identifying hate speech efficiently and accurately irrespective of language is becoming a necessity. Countries like Kenya amongst other African nations have experienced the consequences of not dealing with hate speech as evidenced in previous years. Agencies such as the National Cohesion & Integration Commission were formed to help with this. Section 13 of National Cohesion and Integration Act(2008) outlines what is considered hate speech. In combination with the act an automated way of flagging hate speech would prove helpful for the institution given the country’s context which may not be similar to other countries meaning posts may not be picked/flagged by social media companies such as Twitter and Facebook as a result.
Political hate speech is the greatest area of concern in regards to Kenya and thus we’ll be our area of focus. Looking at whether a post is Hate Speech or Normal Speech. To achieve this, we will leverage state of the art Transformer language models to identify hate speech that strives to be representative of the Kenyan context.


The model is used for classifying a text as Hatespeech or Normal. The model is trained using data from Twitter, specifically Kenyan related tweets. To maximize on the limited dataset, text augmentation was done.

Using a pre-trained "bert-base-uncased" transformer model, adding a dropout layer, a linear output layer and adding 10 common emojis that may be related to either Hate or Nomal Speech. Then the model was tuned on a dataset of Kenyan/Kenyan-related scraped tweets with the purpose of performing text classification of "Normal Speech" or "Hate Speech" based on the text. This model was the result of realizing that majority of similar models did not cater for the African context where the target groups are not based on race and/or religious affiliation but mostly tribal differences which has proved fatal in the past. Additionally, if a tweet is classified as "Hate Speech", zero-shot classification using a mobile bert model was performed with classes: Violent, Offensive and Profane as the candidate labels.

The model can be improved greatly by using a large and representative dataset and optimization of the model to a better degree.


## _Platforms Used_ ##
* Google Colab
* Git/GitHub
* PyCharm


## _Technologies Used_ ##
* Python
* Pandas Library
* Numpy Library
* Matplotlib Library
* Seaborn Library
* Standard Library
* SciPy Library
* Sci-Kit Library
* Plotly
* Pytorch
* Transformers


## _Motivation_ ##
The motivation for this project was to deliver a "Hate Speech" prediction model that was fairly accurate and provided automation for possible institutions such as NCIC Kenya where instead of a person going online to check whether "Hate Speech" related content has been posted then judging it they can use this automated method. 

## _Authors and Acknowledgment_ ##
Assistance: Technical Mentors of Moringa School (Nairobi, Kenya).

## License
MIT © sgich


