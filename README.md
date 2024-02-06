# SentiLens - Uncover reviews' hidden sentiment

## THE BIG IDEA:

Employing aspect-based sentiment analysis (ABSA) to extract valuable feature insights from e-commerce product reviews, thereby empowering consumers to make more informed purchasing decisions and enhancing their overall user experience on the platform.

Utilizing manually annotated reviews for aspect sentiment analysis to extract aspects and predict sentiments from reviews. This enables consumers to obtain a condensed overview of sentiments related to various product features, eliminating the need to delve into an extensive array of reviews. As a result, the decision-making process becomes more streamlined and user-friendly.

## PROPOSED DATA SCIENCE SOLUTION
The project will be structured in phases:

- [X] **Phase 1:** Supervised ABSA
The goal of this phase is to be able to extract all aspect term and their sentment from the given review text. 

- [ ] **Phase 2:** Unsupervised sentiment aspect term extraction
I am hoping to utilize a larger unannotated dataset with rule-based aspect term annotations and re-train the model using this extended dataset in order to improve the model performance and to extend it to more domains.

- [ ] **Phase 3:** App development
The end result of this project is to successfully develop an applications for:
  1. end-users to research e-commerce products' reviews
  2. sellers on the platform to learn more about their sales and customers' feedback
  3. e-commerce platform to improve product listing, seller managements

## THE USER:

The primary beneficiaries of this project encompass two main groups:

**Consumers**:

Who experiences these problems? E-commerce consumers who seek to make well-informed purchasing decisions.

How would they benefit? Consumers often face the challenge of sifting through a multitude of reviews to make a confident purchase. By employing ABSA, this initiative furnishes consumers with a concise and organized overview of sentiment across various product aspects, extracted from fellow consumers’ reviews, enabling them to make more informed decisions efficiently. This translates to a more gratifying shopping experience, reduced uncertainty, and an increased likelihood of making purchases that align with their preferences and needs. Consequently, customers gain a comprehensive understanding of the products they purchase, minimizing the likelihood of encountering unexpected characteristics and subsequently reducing the rate of product returns.

**Sellers & Platform**:

Who experiences these problems? E-commerce platforms and sellers aiming to enhance their sales, reduce return costs, and uphold customer satisfaction.

How would they benefit? E-commerce platforms grapple with challenges such as incomplete product descriptions and limited inventories, hindering their ability to attract customers and minimize returns. By integrating ABSA, the outcomes of this project can assist platforms in refining product descriptions based on identified consumer preferences and concerns. This leads to an enriched online marketplace experience, higher sales conversion rates. Additionally, the enhanced knowledge gained by customers about their purchases directly contributes to fewer unexpected characteristics, resulting in a decreased rate of product returns.


## THE IMPACT: 

The anticipated impact of our project extends both to societal and business dimensions, with quantifiable benefits that directly address critical challenges in the e-commerce landscape:

**Business Value**:

Increased Profitability: Our project aims to increase e-commerce sales and lower return rates. 
According to CNBC, the online purchase return rate in 2021 was 21%, which represents a substantial financial burden for e-commerce platforms. By empowering consumers to make more informed purchasing decisions through our ABSA, we anticipate a substantial reduction in returns. For instance, if our approach reduces the return rate by just 5%, this could lead to potential savings of over $20 billion at Amazon, based on Amazon’s 2021 net sales revenue.

Improved Customer Experience: With reviews being a top influence on consumer behavior and purchasing decisions  where 77% of consumers reported consistently or frequently consult reviews prior to purchases, our project directly addresses the consumer frustration arising from irrelevant search results and time-consuming navigation through extensive review content.  Based on a survey, 60% of shoppers get annoyed by searches that don't match what they're looking for, and almost half (47%) say it takes too long to find what they want, while 41% struggle to find the exact thing they need. By enhancing the product descriptions and facilitating better understanding of reviews, consumers can more efficiently locate and acquire the exact products they desire. This leads to heightened customer satisfaction and increased loyalty.

Logistics Cost Reduction: E-commerce giants like Amazon spend significant resources on logistics, with returns being a substantial contributor to these costs. By leveraging our approach to minimize returns, platforms can potentially save a substantial portion of their logistics expenses. For instance, a 5% reduction in return-related logistics costs could translate to savings of approximately $7.6 billion, based on Amazon's 2021 logistics spending.

**Societal Value**:

Reduced landfill waste: The prevalence of returned products contributing to landfill waste is a concerning environmental issue. By implementing our ABSA approach to minimize returns, the amount of unsellable items destined for landfills can be significantly reduced. This translates into a positive ecological impact by curbing unnecessary waste production.

Decreased CO2 Footprint: Reverse delivery processes, required for returned items, contribute to higher carbon emissions due to transportation. With a decrease in product returns through our approach, the need for reverse logistics and subsequent carbon emissions can be mitigated, resulting in a tangible reduction in CO2 footprint.

## THE DATA:
The data being used in this project is obtained from <a href='https://huggingface.co/datasets/jakartaresearch/semeval-absa'>Hugging Face SemEval - SemEval-2015 Task 12</a>
This dataset contains annotated review text from Amazon Laptop.
- Dataset size: 3.85k rows
- Data structure:
  - id: int, unique identifier of each text
  - text: str, content of the text
  - aspects: dict{term: array str, polarity: array str, from: array int, to: array int}
    Annotated aspects for each text
    - term: aspect extracted from text, each text may contain 0 to more than 1 terms
    - polarity: sentiment for each term in text with values: positive, negative, neutral and conflict
    - from: the start index of the identified term
    - to: the end index of the identified term

## DATA PREPARATION
The data is transformed to a word token based and aspects are labelled using unified BIO technique (as introduced by <a href='https://aclanthology.org/2020.emnlp-main.453.pdf'>Wu et al., 2020</a>) which combines aspect boundaries and aspect sentiment.
  Word boundaries:
  - B: indicates the 1st word in the aspect term
  - I: indicates the subsequent word in the aspect term
  - O: indicates words that are not part of any aspect term

  Aspect sentiment:
  - POS: positive
  - NEU: neutral
  - NEG: conflict

This unified BIO label technique is more effective in recognizing unigram and n-gram aspect terms comparing to a binary classification (whether a token is part of an aspect). By using a unified a approach, we can combine two tasks: aspect extraction and sentiment classification into one task.

## EDA:
  ### 1. Word counts per review sentence
  ### 2. Aspect vs non-aspect distribution
  ### 3. Part of speech & aspects
  ### 4. Aspect polarity & context sentiments


## PERFORMANCE METRICS:
  Implemented models are evaluated using CoNLL F1 score (as described by <a href='https://aclanthology.org/W02-2024/'>Tjong Kim Sang, 2002</a>) which is a much more restricted f1 score that is designed for token classification (named entity classification) models. The score only give credits to "whole" aspect accuracy, partial matchings are considered a failure in this metric.

## MODELS:
This project includes application of fundamental machine learning algorithms (random forest & CRF) and advanced transfer learning neural network from pretrained model (DistilBERT).

### Fundamental ML algorithms:

I have created the below features for each token (word) in the sentence, such as:
- word (the word itself)
- stemming / lemming versions of word
- part of speech (POS) of word
- words sentiment lexicon
- context words within a pre-defined window (5 words surrounding the token)
- context words stemming/ lemming
- context words POS
- context words sentiment lexicon
- ...

The list is not exhaustive, and is an iterative process as we perform EDA and go back and refining/ adding more features.

Overall results for both a default random forest and CRF model were coNLL f1 = 0, while token only f1 score (macro) was 0.14. This suggested that the provided features did not provide a lot of information for the models to train on, and the models failed the most is in recognizing the aspect's polarity.

## INSTALLATION
- Tested on Python 3.9.16 (recommended to use a virtual environment such as Pyenv)
- Install data and requirements: bash setup.sh


