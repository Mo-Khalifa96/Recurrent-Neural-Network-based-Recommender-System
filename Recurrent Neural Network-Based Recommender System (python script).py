#Importing the modules for use 
import os 
import re
import math
import requests
import textwrap
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform, pdist, jaccard
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import stopwordsiso as stopwords
from langdetect import detect
import stanza
import tensorflow as tf
from tensorflow.keras import layers, optimizers  
from tensorflow.keras.preprocessing.text import Tokenizer    
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping 


import warnings
warnings.simplefilter('ignore')   #disable python warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   #disable tensorflow warnings

#Adjust pandas data display settings 
pd.set_option('display.max_colwidth', 100)
#Set plotting context 
sns.set_context('paper') 

#Set random seed for reproducible results
rs = 252

#set global seed for numpy and tensorflow
np.random.seed(rs)
tf.random.set_seed(rs)


#Define function to display books by their covers
def get_covers(books_df: pd.DataFrame):
    n_books = len(books_df.index)
    n_cols = ((n_books + 1) // 2) if n_books > 5 else n_books
    n_rows = math.ceil(n_books / n_cols)
    
    #create figure and specify subplot characeristics
    plt.figure(figsize=(4.2*n_cols, 6.4*n_rows), facecolor='whitesmoke')
    plt.subplots_adjust(bottom=.1, top=.9, left=.02, right=.88, hspace=.32)  
    plt.rcParams.update({'font.family': 'Palatino Linotype'})   #adjust font type

    #request, access and plot each book cover 
    for i in range(n_books):
        try:
            response = requests.get(books_df['cover_image_uri'].iloc[i])
        except:
            print('\nCouldn\'t retrieve book cover. Check your internet connection and try again...\n\n', flush=True)
            return
        
        #access and resize image
        img = Image.open(BytesIO(response.content))
        img = img.resize((600, 900))
        
        #shorten and wrap book title
        full_title = books_df['book_title'].iloc[i]
        short_title = re.sub(r'[:?!].*', '', full_title)
        title_wrapped = "\n".join(textwrap.wrap(short_title, width=26))
        
        #plot book cover 
        plt.subplot(n_rows, n_cols, i+1)
        plt.imshow(img)
        plt.title(title_wrapped, fontsize=21, pad=15)
        plt.axis('off')    
    plt.show()


#Define custom function to visualize model training history
def plot_training_history(run_histories: list, metrics: list = [None], title='Model run history'):
    #If no specific metrics are given, infer them from the first history object
    if metrics is None:
        metrics = [key for key in run_histories[0].history.keys() if 'val_' not in key]
    else:
        metrics = [metric.lower() for metric in metrics]

    #Set up the number of rows and columns for the subplots
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)  #Limit to a max of 3 columns for better readability
    n_rows = math.ceil(n_metrics / n_cols)

    #Set up colors to use
    colors = ['steelblue', 'red', 'skyblue', 'orange', 'indigo', 'green', 'DarkCyan', 'olive', 'brown', 'hotpink']

    #Ensure loss first is plotted first
    if 'loss' in metrics:
        metrics.remove('loss')
        metrics.insert(0,'loss')

    #Initialize the figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5*n_cols, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]

    #Loop over each metric and create separate subplots
    for i, metric in enumerate(metrics):
        #Initialize starting epoch
        epoch_start = 0
        for j, history in enumerate(run_histories):
            epochs_range = range(epoch_start, epoch_start + len(history.epoch))

            #Plot training and validation metrics for each run history
            axes[i].plot(epochs_range, history.history[metric], color=colors[i*2], ls='-', lw=2, label=(f'Training {metric}') if j==0 else None)
            axes[i].set_xticks(epochs_range)
            if f'val_{metric}' in history.history:
                axes[i].plot(epochs_range, history.history.get(f'val_{metric}', []), color=colors[i*2+1], ls='-', lw=2, label=(f'Validation {metric}') if j==0 else None)

            #Update the epoch start for the next run
            epoch_start += len(history.epoch)

        #Set the titles, labels, and legends
        axes[i].set(title=f'{metric.capitalize()} over Epochs', xlabel='Epoch', ylabel=metric.capitalize())
        axes[i].legend(loc='best')

    #Remove any extra subplots if the grid is larger than the number of metrics
    for k in range(i + 1, n_rows * n_cols):
        fig.delaxes(axes[k])

    fig.suptitle(title, fontsize=16, y=(0.95) if n_rows>1 else 0.98)
    #plt.tight_layout(pad=1.1)  # (left, bottom, right, up)
    plt.show()



# Part One: Reading and Inspecting the Data
#Access and read data into dataframe
df = pd.read_csv('Book_Details.csv', index_col='Unnamed: 0')

#drop unnecessary columns 
df = df.drop(['book_id', 'format', 'authorlink', 'num_pages'], axis=1)


#report the shape of the dataframe
shape = df.shape
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0])


#Preview first 5 entries
df.head()

#Inspect coloumn headers, data type, and number of entries
df.info()


#get overall description of object columns
print(df.describe(include='object').T)
print('\n'+ 80*'_' +'\n')

#get statistical summary of the numerical data
print(df.describe().drop(['25%', '50%', '75%']).apply(lambda x: round(x)).T)



# Part Two: Cleaning and Updating the Data
#first, normalize book titles by removing punctuation
df['normalized_title'] = df['book_title'].apply(lambda title: re.sub(r'[^\w\s]', '', title))

#drop duplicate book titles and reset dataframe index
df = df.drop_duplicates(subset='normalized_title', ignore_index=True)


#check the number of books with inappropiate book description or NaN (not a number) values
print('Number of entries with NaN values in the book details column (before): ', df['book_details'].isna().sum())

#fill NaN book details with empty strings
df['book_details'] = df['book_details'].fillna('')

#check the number of entries after 
print('\nNumber of entries with NaN values in the book details column (after): ', df['book_details'].isna().sum())


#Changing string list to list then to string with the genres of books
df['genres'] = df['genres'].apply(lambda x: ', '.join(eval(x)))

#Updating rows with no genre
#get indices of books with no genre labels
no_genre_before = df[df['genres'].str.len() == 0].index

#we can preview the books identified
df.iloc[no_genre_before, 1:8].head(3)


#Get total number of books with no genre before the update
print('Total number of entries with missing genre (before): ', len(df.iloc[no_genre_before]))

#change empty strings with genres common to given author
for i in no_genre_before:
    genre_labels = df[df['author']==df['author'].iloc[i]]['genres'].iloc[0]
    if len(genre_labels) > 0:
        df.at[i, 'genres'] = genre_labels
    else:
        df.drop(index=i, inplace=True)
#resetting dataframe index
df.reset_index(drop=True, inplace=True)

#check number of books with no genre after the update
no_genre_after = df[df['genres'].str.len() == 0].index
print('\nTotal number of entries with missing genre (after): ', len(df.iloc[no_genre_after]))


#create empty list for storing indices of books with conflicting genres and set count to zero
indices=[]
count=0
#loop over and return all books with conflicting genres
for genre_string, title in zip(df['genres'], df['book_title']):
    if 'Fiction' in genre_string and 'Nonfiction' in genre_string:
        count += 1
        indices.append(df[df['book_title']==title].index)
        print(f'{count}. {title} // {genre_string}')


#create dictionary with sub-strings to be replaced or removed
replacements_dict = { 'Military Fiction': 'Military',
                      'Literary Fiction': 'Literary',
                      'Realistic Fiction': 'Realistic',
                      'Non Fiction': 'Nonfiction' }

#replace substrings according to specified values
df['genres'] = df['genres'].replace(replacements_dict, regex=True)

#Now we can check again
count=0
for genre_string, title in zip(df['genres'], df['book_title']):
    if 'Fiction' in genre_string and 'Nonfiction' in genre_string:
        count += 1
print(f'Number of books with conflicting genres:  {count}')


#Changing string list in publication info column to normal string
df['publication_info'] = df['publication_info'].apply(lambda x: eval(x)[0] if len(eval(x)) > 0 else 'n.d.')

#extract year of publication from publication info column and assign it to a new data column, 'publication_year' (if 'n.d.' assign an empty string)
df['publication_year'] = df['publication_info'].str.extract(r'(\d{1,4}$)').fillna('')

#preview changes and new publication year column
df[['publication_info', 'publication_year']].sample(5)


#Create new column for book's language 
def get_language(idx):
    try:
        #detect language of book details 
        return detect(df['book_details'].iloc[idx])
    except:
        #infer from the book title
        return detect(df['book_title'].iloc[idx])
    
df['language'] = [get_language(idx) for idx in range(len(df.index))]

#preview sample 
print(df[['book_title', 'language']].sample(5))
print()

#Report number of languages in the dataset 
print('Number of languages featured in the dataset: ', len(df['language'].unique()))
print()

#plot the distribution of non-english books in the dataset
plt.bar(df['language'].value_counts().index[1:], df['language'].value_counts().values[1:], color='#24799e')
plt.title('Disribution of non-english languages', fontsize=12)
plt.xticks(rotation=90, fontsize=8.8)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



# Part Three: Exploratory Data Analysis
#Create one-hot encoded dataframe with all unique genres in the data
genres_df = df['genres'].str.get_dummies(', ').astype(int)

#preview genres dataframe
genres_df.head()


#Extract top 20 genres by genre frequency
top20_genres = genres_df.sum().sort_values(ascending=False)[:20]

#Visualize top 20 genres using bar chart
top20_genres.plot(kind='bar', color='#24799e', width=.8, figsize=(7.5,5),
                    linewidth=.8, edgecolor='k', rot=90)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()

#Assign appropriate data type to the rating distribution column
df['rating_distribution'] = df['rating_distribution'].apply(lambda x: eval(x))

#get total number of five star ratings per book from the rating distribution column
df['total_5star_ratings'] = [int(dic['5'].replace(',','')) for dic in df['rating_distribution']]

#sort data by books with highest frequency of 5 star ratings
top10_books = df.sort_values(by='total_5star_ratings', ascending=False).iloc[:10][['book_title', 'author', 'genres', 'cover_image_uri']].reset_index(drop=True)

#report the results table
top10_books.iloc[:,:3]


#get and display books by cover
get_covers(top10_books)


#Aggregate ratings by rating star 
rating_counts = {'5':0, '4':0, '3':0, '2':0, '1':0}
for ratings in df['rating_distribution']:
    for key, value in ratings.items():
        rating_counts[key] += int(value.replace(',',''))

#plot the ratings frequency distribution
plt.figure(figsize=(7.5,5))
plt.bar(rating_counts.keys(), rating_counts.values(), color='#24799e', width=.7, linewidth=.8, edgecolor='k')
plt.title('Frequency Distribution of Star Ratings', fontsize=11)
plt.xlabel('Star Rating', fontsize=10)
plt.ylabel('Frequency of Rating', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=.7)
plt.show()


#Visualize the relationship between the number of ratings and the average rating 
# score for a given book using scatter plot
plt.figure(figsize=(9,5))
sns.scatterplot(data=df, x='num_ratings', y='average_rating')
plt.gcf().axes[0].xaxis.get_major_formatter().set_scientific(False)
plt.xticks(rotation=-30)
plt.title('Relationship between Number of Ratings and Average Rating', fontsize=13)
plt.xlabel('Number of Ratings', fontsize=11.5)
plt.ylabel('Average Book Rating', fontsize=11.5)
plt.show()



# Part Four: Text Preprocessing
#Combine features for ovarall text processing
df['combined_features'] = (df['book_title'] + ' / ' + df['author'] + ' / ' + df['publication_year'] + ' / ' + df['genres'] + ' / ' + df['book_details'])

#preview a sample of the combined features column
for row in df['combined_features'][10:15]:   #book 10 to 15
    print(row[:200],'\n')


books_data = df['combined_features']   #I will now use this going forward

#Remove punctuations and normalize text 
books_data = books_data.apply(lambda text: ' '.join(re.findall(r'\b\w+\b', text.lower().strip())))

#preview sample 
books_data[10:15] 

#Create dictionary for storing language-stopwords pairs
stopwords_multilang = {lang: stopwords.stopwords(lang) for lang in stopwords.langs()}

#Define function to remove stop words for text of a given language 
def remove_stopwords(text,  stopwords_multilang, language=None):
    if language is None:
        language = detect(text)
    filtered_text = [word for word in text.split() if word not in stopwords_multilang.get(language,{})]
    return ' '.join(filtered_text)

#Remove stop words 
books_data = pd.Series([remove_stopwords(books_data[i], stopwords_multilang, language=df['language'].iloc[i]) for i in range(len(books_data))])
books_data[10:15] 


#First, I will create a dictionary with the languages in the dataset
lang_dict = {}
lang_dict = lang_dict.fromkeys(df['language'].unique())

#Assign a lematization model for each language separately, using nltk for english and stanza for non-english
for lang in list(lang_dict.keys())[1:]:
    try:
        #assign model if the language is supported 
        lang_dict[lang] = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma', verbose=0)
    except:
        lang_dict[lang] = None

#get supported languages 
supported_langs = [key for key,val in lang_dict.items() if val is not None]
print('Number of languages supported:', len(supported_langs)+1)
print('Number of languages not supported:', len(lang_dict)-len(supported_langs)-1)


#Initiate english lemmatizer 
en_lemmatizer = WordNetLemmatizer()

#Define function to lemmatize text
def lemmatize_text(text, language=None):
    if language is None:
        language = detect(text)
    #nltk is best for english 
    if language=='en':
        text = [en_lemmatizer.lemmatize(word, pos='v') for word in text.split()]
        text = [en_lemmatizer.lemmatize(word, pos='r') for word in text]
        return ' '.join([en_lemmatizer.lemmatize(word, pos='n') for word in text])
    #otherwise, use stanza if language is supported
    elif language in supported_langs:  
        nlp = lang_dict[language]
        doc = nlp(text).iter_words()
        return ' '.join([word.lemma if word.upos in ('ADV', 'NOUN', 'VERB') else word.text for word in doc])
    else:
        return text 

#Lemmatize the books 
books_data = pd.Series([lemmatize_text(books_data[i], language=df['language'].iloc[i]) for i in range(len(books_data))])

# #preview sample 
books_data[10:15] 


#Tokenize text
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(books_data)

#get indices per tokens and report vocabulary size 
word2idx = tokenizer.word_index 
idx2word = {idx: word for word, idx in word2idx.items()}
vocab_size = len(word2idx) + 1 
print('vocabulary size:', vocab_size)
print()

#convert the text into sequences of word indice  
books_data = tokenizer.texts_to_sequences(books_data)

#Confirm if the words and tokenized correctly using the idx2word dictionary 
#decode sample book description 
print([word_idx for word_idx in books_data[10]][:20])
print(' '.join([idx2word[word_idx] for word_idx in books_data[10]][:20]))

#Check the sequence length distribution in the data
seq_lengths = [len(seq) for seq in books_data]

#show distribution of book description lengths
perc = np.percentile(seq_lengths, 95)
sns.histplot(seq_lengths, bins=100, kde=True)
plt.title('Distribution of Book Description Lengths')
plt.axvline(perc, linestyle='--', color='lightgray', linewidth=1, label=f'95th percentile: {perc:.2f}')
plt.text(perc*1.2, plt.gca().get_ylim()[1] * 0.85, f'{perc:.1f}\n(95th percentile)', rotation=30, color='gray', ha='center', fontsize=7)
plt.show()

#Now, I will identify the maximum sequence length for padding as the 95th percentile
max_seq_len = int(np.percentile(seq_lengths, 95))

#Sequence Padding
books_data = pad_sequences(books_data, maxlen=max_seq_len, padding='post', truncating='post')

#Report data shapes after padding
print('Books data shape:', books_data.shape)



# Part Five: Model Development and Training
#Preparing Training Data and Loss Function
#prepare books descriptions for TF-IDF vectorization
book_descriptions = []
for seq in books_data: 
    #exclude 0s (the padding) and convert tokens back to words
    words = [idx2word[token] for token in seq if token != 0]
    book_descriptions.append(' '.join(words))

#Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

#fit and transform the books descriptions to get a TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(book_descriptions)

#Now compute cosine similarity on the TF-IDF matrix
similarity_matrix = cosine_similarity(tfidf_matrix)

#Get a statistical summary of the similarity matrix 
mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
q25, q50, q75 = np.percentile(similarity_matrix[mask], [25, 50, 75]) 
print(f'Similarity range: {np.max(similarity_matrix):.5f} - {np.min(similarity_matrix):.5f}')
print("25th percentile:", q25.round(4))
print('50th percentile:', q50.round(4))  
print("75th percentile:", q75.round(4))
print("IQR:", (q75 - q25).round(4))


#Preparing triplets for training
#create list to store triplets' indices 
triplets_indices = []

#Controlling for outliers 
avg_similarities = np.mean(similarity_matrix, axis=1)
outliers = np.where(avg_similarities > np.percentile(avg_similarities, 99))[0]

#Loop over each anchor sample and store triplets' indices
for anchor_idx in range(len(books_data)):
    if anchor_idx in outliers: 
        continue
    similarity_scores = similarity_matrix[anchor_idx]
    similarity_scores[anchor_idx] = -np.inf  #ignore self-similarity
    
    #Specify threshold for selection of positive samples 
    pos_threshold = np.percentile(similarity_scores, 95)   #to sample from top 5% similarity range 

    #specify triplets mining margin for negative samples 
    hard_mining_margin = np.percentile(similarity_scores, 95) - np.percentile(similarity_scores, 85)   #to sample from the 10% similarity range below the positives 
    soft_mining_margin = np.percentile(similarity_scores, 95) - np.percentile(similarity_scores, 70)    #to sample from the 25% similarity range below the positives 


    #Specify range for positives and obtain positive sample (from the top 5%)
    positives_range = np.where(similarity_scores >= pos_threshold)[0]  
    if len(positives_range) == 0: 
        continue 
    positive_idx = np.random.choice(positives_range)
    positive_scores = similarity_scores[positive_idx]

    #Specify range for negatives and obtain negative sample (either from the 10% or 25% similarity range below the positives)
    negatives_range = np.where((similarity_scores < positive_scores) & (similarity_scores >= positive_scores - hard_mining_margin))[0]
    if len(negatives_range) == 0:
        negatives_range = np.where((similarity_scores < positive_scores) & (similarity_scores >= positive_scores - soft_mining_margin))[0]
        if len(negatives_range) == 0:
            continue
    negative_idx = np.random.choice(negatives_range)
    
    #append triplets 
    triplets_indices.append((anchor_idx, positive_idx, negative_idx))

#convert to numpy array and report number of generated triplets 
triplets_indices = np.array(triplets_indices)
print(f"\nGenerated {len(triplets_indices)} triplets using cosine similarity")


#Create triplets dataset for training 
triplets_dataset = tf.data.Dataset.from_tensor_slices(({
        'anchor_input': books_data[triplets_indices[:, 0]],
        'positive_input': books_data[triplets_indices[:, 1]],
        'negative_input': books_data[triplets_indices[:, 2]]}, 
        np.zeros((len(triplets_indices),128)))) 

#set batch size and enable prefetching 
triplets_dataset = triplets_dataset.batch(64).prefetch(tf.data.AUTOTUNE)


#Define custom triplet loss for semi-hard triplets
def triplet_loss(margin=1.0):
    def loss(y_true, y_pred):
        #Get triplets' embeddings 
        anchor_embeddings, positive_embeddings, negative_embeddings = y_pred[0], y_pred[1], y_pred[2]

        #Calculate mean squared distances
        pos_distance = tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=-1)
        neg_distance = tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=-1)

        #Calculate loss (with margin constraint)
        basic_loss = pos_distance - neg_distance + margin 
        return tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss


# Model Development 
#Define embeddings dimensions 
embedding_dims = 300 

#Create embeddings matrix using GloVe 
#build embeddings index from GloVe file
embeddings_index = {}
with open('glove.840B.300d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector_values = values[1:]
        if len(vector_values) > embedding_dims:
            vector_values = vector_values[-embedding_dims:]
        coefs = np.asarray(vector_values, dtype='float32')
        embeddings_index[word] = coefs

#Create embedding matrix 
embedding_matrix = np.zeros((vocab_size, embedding_dims))
for word, idx in word2idx.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector


#Define self-attention layer 
class SelfAttentionLayer(layers.Layer):
    def call(self, inputs):
        #query, key, value 
        Q, K, V = inputs, inputs, inputs     
        #scaling factor 
        d_k = tf.cast(tf.shape(K)[-1], tf.float32) 
        #compute attention weights 
        attention_weights = tf.nn.softmax(tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k), axis=-1)
        return tf.matmul(attention_weights, V)  #attention vectors


#Define model subclass to build and train a triplet recurrent neural network 
class Triplet_LSTM_Model(tf.keras.Model):
    def __init__(self, input_dims, embedding_dims=300, vocab_size=50000, LSTM_units=128, dense_units=(128,64), **kwargs):
        '''
        :param int input_dims: Number of input dimensions. Positional parameter.
        :param int embedding_dims: Number of embedding dimensions for the embedding layer. Default is 300 dims.
        :param int vocab_size: Size of the input vocabulary for the embedding layer. Default is 50,000 words.
        :param int LSTM_units: Number of units for the bidirectional LSTM layer. Default is 128 units.
        :param tuple dense_units: Number of units for the two dense layers following the attentional layer. Default is (128,64).
        '''

        super().__init__(**kwargs)
        #Initialize parameters 
        self.input_dims = input_dims 
        self.embedding_dims = embedding_dims 
        self.embedding_input = vocab_size 
        self.LSTM_units = LSTM_units 
        self.dense_units = dense_units
        self.name = 'Triplet_LSTM_Model'        
        #Initialize attention layer and models used
        self.Attention_layer = SelfAttentionLayer(name='SelfAttention_layer')
        self.LSTM_network = self._build_LSTM_network()
        self.Triplet_Model = self._build_triplet_model()


    def _build_LSTM_network(self):
        #Build LSTM model 
        model_inputs = layers.Input(shape=(self.input_dims,), name='Input_layer')
        x = layers.Embedding(input_dim=self.embedding_input, output_dim=self.embedding_dims, weights=[embedding_matrix], 
                             trainable=True, mask_zero=True, name='Embedding_layer')(model_inputs)
        x = layers.Bidirectional(
            layers.LSTM(self.LSTM_units, activation='tanh', kernel_initializer='lecun_uniform', 
            return_sequences=True, recurrent_initializer='orthogonal', name='LSTM_layer'), name='Bidirectional_layer')(x)
        x = layers.LayerNormalization(epsilon=1e-6, name='LayerNorm_post_LSTM')(x)
        x = self.Attention_layer(x)
        x = layers.GlobalAveragePooling1D(name='GlobalAvgPooling1D')(x)
        x = layers.Dense(self.dense_units[0], activation='relu', kernel_initializer='he_uniform', name='Dense_layer1')(x)
        x = layers.LayerNormalization(epsilon=1e-6, name='LayerNorm_post_Dense1')(x)
        model_outputs = layers.Dense(self.dense_units[1], activation='relu', kernel_initializer='he_uniform', name='Dense_layer2')(x)
        return tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name='LSTM_model')


    def _build_triplet_model(self):
        #Build triplets model 
        base_model = self.LSTM_network
        #Define input layers
        anchor_input = layers.Input(shape=(self.input_dims,), name='anchor_input')
        positive_input = layers.Input(shape=(self.input_dims,), name='positive_input')
        negative_input = layers.Input(shape=(self.input_dims,), name='negative_input')
        #Compute embeddings for each of the triplet
        anchor_output = base_model(anchor_input)
        positive_output = base_model(positive_input)
        negative_output = base_model(negative_input)

        #Build and return triplet model
        return tf.keras.Model(
                inputs=[anchor_input, positive_input, negative_input],
                outputs=[anchor_output, positive_output, negative_output],
                name='Triplet_Model')

    def call(self, inputs, training=None):
        #Model fitting function
        #Get anchor, positive, and negative inputs 
        #Handle dictionary inputs
        if isinstance(inputs, dict):
            anchor_input = inputs['anchor_input']
            positive_input = inputs['positive_input']
            negative_input = inputs['negative_input']
        else:
            #Handle list/tuple inputs
            anchor_input, positive_input, negative_input = inputs
        
        #Obtain and return triplets embeddings (y_pred)
        triplets_embeddings = self.Triplet_Model([anchor_input, positive_input, negative_input])
        return triplets_embeddings



# Model Training 
#Define model parameters 
input_dims = books_data.shape[1]   #sequence length
embedding_dims = 300
LSTM_units = 128 
dense_units = (256,128)

#Triplets model 
model = Triplet_LSTM_Model(input_dims=input_dims, embedding_dims=embedding_dims, vocab_size=vocab_size, LSTM_units=LSTM_units, dense_units=dense_units)

#Compile the model 
model.compile(optimizer=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.95, clipnorm=5.0), loss=triplet_loss(margin=4.0))

#Initialize learning rate schedule for the optimizer
reduceOnPleateau_lr = ReduceLROnPlateau(monitor='loss', mode='min', factor=0.8, min_delta=0.001, patience=4, min_lr=1e-6, verbose=1) 

#Define early stopping criterion
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=8, start_from_epoch=10, restore_best_weights=True)

#Train the model
run_history = model.fit(triplets_dataset, epochs=25, batch_size=64, callbacks=[reduceOnPleateau_lr, early_stop], verbose=1)

#Visualize model's run history 
plot_training_history([run_history], ['loss'], 'LSTM model run history') 


#Get embeddings model from the larger model trained with the triplets
embeddings_model = model.LSTM_network

#Save the final embeddings model
embeddings_model.save('embeddings_model.keras')

#load model with the custom attention layer 
#embeddings_model = tf.keras.models.load_model('embeddings_model.keras', custom_objects={'SelfAttentionLayer': SelfAttentionLayer})

#Get book embeddings 
book_embeddings = embeddings_model.predict(books_data)

#Compute cosine similarity on the embeddings for overall book similarity
overall_similarity_mtrx = cosine_similarity(book_embeddings)


#Convert genres_df to CSR matrix
genres_csr_mtrx = csr_matrix(genres_df.values).astype(bool).toarray()

#Compute jaccard distance similarity and return jaccard similarity matrix
genre_sim_mtrx = 1 - squareform(pdist(genres_csr_mtrx, metric=jaccard))

#normalize jaccard distance scores
genre_sim_mtrx = genre_sim_mtrx / np.max(genre_sim_mtrx) if np.max(genre_sim_mtrx) > 0 else genre_sim_mtrx



# Part Six: Building a Book Recommendation Function
#Define helper functions to return book recommendations
def Get_Recommendations(title: str, overall_sim_mtrx: np.ndarray, genre_sim_mtrx: np.ndarray, alpha=0.5, top_n=10):
    '''
    This function takes a book title and recommends similar books that cover similar themes
    or fall within the same genre categories.
    
    Parameters:
    - title (str): The title of the book for which recommendations are sought. 
    - overall_sim_mtrx (ndarray): A similarity matrix based on book overall similarities, where each row 
      corresponds to a book and each column corresponds to its cosine similarity score with other books.
    - genre_sim_mtrx (ndarray): A similarity matrix based on book genres, where each row
      corresponds to a book and each column corresponds to its jaccard similarity score with
      other books based on genre.
    - alpha (float, optional): Weighting factor for combining overall similarity and genre
      similarity. Defaults to 0.5, balancing overall similarity and genre similarity together.
    - top_n (int, optional): Number of recommendations to return. Defaults to 10.
    
    Returns:
    - Data table (Series) with recommended books and plot of each book with its cover.

    Raises:
    - TypeError: If the title provided is not a string.

    Notes:
    - This function filters, preprocesses and standardizes the book titles given, identifies its genre
      categories, importantly, identifying whether it's Fiction or Nonfiction work to prevent genre
      overall while looking for recommendations.
    - It looks for book recommendations by combining similarity scores from two matrices: overall_sim_mtrx
      (based on overall similarities) and genre_sim_mtrx (based on genre similarity).
    - It prioritizes books with similar genre categories; otherwise, it recommends book based on
      overall book similarity. However, the degree of each's influence can be controlled with the alpha parameter.
    - Finally, recommendations are filtered to include books by a different variety of authors, limiting
      the number of recommendations to only 5 books per one author.
    - The number of book recommendations can be adjusted using the 'top_n' parameter. Returns 10 book recommendations by default.
    '''

    #check if title provided is of the correct data type (string)
    try:
        curr_title = str(title)
    except:
        raise TypeError('Book title entered is not string.')

    #standardize titles for accurate comparisons
    title = curr_title.lower().strip()
    full_titles = df['book_title'].apply(lambda title: title.lower().strip())
    partial_titles = full_titles.str.extract(r'^(.*?):')[0].dropna()

    #check if provided title matches book title in the dataset and get index if found
    if title in full_titles.values:
        idx = df[full_titles == title].index[0]

    elif title in set(partial_titles.values):
        idx_partial = partial_titles[partial_titles == title].index[0]
        idx = df[df['book_title'] == df['book_title'].iloc[idx_partial]].index[0]

    else:
        #try normalizing book titles across the board by removing punctuations and performing some stemming on them
        normalized_title = re.sub(r'(^\s*(the|a)\s+|[^\w\s])', '', title, flags=re.IGNORECASE)
        normalized_title = re.sub(r'\b(\w+?)(s|ing)\b', r'\1', normalized_title, flags=re.IGNORECASE)
        normalized_full_titles = full_titles.apply(lambda title: re.sub(r'(^\s*(the|a)\s+|[^\w\s])', '', title, flags=re.IGNORECASE))
        normalized_full_titles = normalized_full_titles.apply(lambda title: re.sub(r'\b(\w+?)(s|ing)\b', r'\1', title, flags=re.IGNORECASE))
        normalized_partial_titles = partial_titles.apply(lambda title: re.sub(r'(^\s*(the|a)\s+|[^\w\s])', '', title, flags=re.IGNORECASE))
        normalized_partial_titles = normalized_partial_titles.apply(lambda title: re.sub(r'\b(\w+?)(s|ing)\b', r'\1', title, flags=re.IGNORECASE))
        #check title match
        if normalized_title in set(normalized_full_titles.values):
            idx = df[normalized_full_titles == normalized_title].index[0]

        elif normalized_title in set(normalized_partial_titles.values):
            idx_partial = normalized_partial_titles[normalized_partial_titles==normalized_title].index[0]
            idx = df[df['book_title'] == df['book_title'].iloc[idx_partial]].index[0]

        else:
            print(f'\nBook with title \'{curr_title}\' is not found. Please try a different book.\n', flush=True)
            return False


    #Check if 'Fiction' is in the genre of the selected book
    is_fiction = 'Fiction' in df['genres'].iloc[idx]

    #Find books with the same genre category
    if is_fiction:
        book_indices_ByGenre = [i for i in df.index if ('Fiction' in df['genres'].iloc[i]) and (i != idx)]
    else:
        book_indices_ByGenre = [i for i in df.index if ('Fiction' not in df['genres'].iloc[i] or 'Nonfiction' in df['genres'].iloc[i]) and (i != idx)]

    #Filter books to include books written in the same language as the target book
    book_indices_final = [i for i in book_indices_ByGenre if df['language'].iloc[i] == df['language'].iloc[idx]]
    
    #if empty, fallback to indices by genre 
    if not book_indices_final:
      book_indices_final = book_indices_ByGenre  


    #Combine the two similarity matrices using weighted sum
    weighed_similarity = (alpha * overall_sim_mtrx[idx]) + ((1 - alpha) * genre_sim_mtrx[idx])

    #Get cosine similarity scores for books with the same genre
    similarity_scores = [(i, weighed_similarity[i]) for i in book_indices_final]

    #Filter scores to only include books with the same genre (and language) category
    similarity_scores = [score for score in similarity_scores if score[0] in book_indices_final]

    #Sort the books based on the genre similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    #If less than top_n books are found in the same genre category, add books by closest overall cosine distance
    if len(similarity_scores) < top_n:
        cos_scores = list(enumerate(weighed_similarity[idx]))
        cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse=True)
        cos_scores = [score for score in cos_scores if score[0] != idx and score[0] not in [x[0] for x in similarity_scores]]  #exclude the same book and already recommended books
        similarity_scores += [score for score in cos_scores if score not in similarity_scores][:top_n - len(similarity_scores)]  #add books until reaching top_n

    #Limit recommendations to 5 books per author
    author_counts = {}
    similarity_scores_filtered = []
    for score in similarity_scores:
        author = df['author'].iloc[score[0]]
        if author not in author_counts or author_counts[author] < 5:
            similarity_scores_filtered.append(score)
            author_counts[author] = author_counts.get(author, 0) + 1


    #Get the scores of the N most similar books
    most_similar_books = similarity_scores_filtered[:top_n]
    #Get the indices of the books selected
    most_similar_books_indices = [i[0] for i in most_similar_books]

    #Prepare DataFrame with recommended books and their details
    recommended_books = df.iloc[most_similar_books_indices][['book_title', 'author', 'cover_image_uri']]
    recommended_books['Recommendation'] = recommended_books.apply(lambda row: f"{row['book_title']} (by {row['author']})", axis=1)
    recommended_books['Genre'] = df.iloc[most_similar_books_indices]['genres'].apply(lambda x: x.split(', ')[0])
    recommended_books.reset_index(drop=True, inplace=True)

    #Return book recommendations
    print(f"\nRecommendations for '{curr_title.title()}' (by {df['author'].iloc[idx]}):", flush=True)
    print(recommended_books[['Recommendation','Genre']].rename(lambda x:x+1,axis=0))
    print('\n', flush=True)
    get_covers(recommended_books)
    return



# Part Seven: Testing the Recommendation System
#Adjust pandas display settings to display entire column
pd.set_option('display.max_colwidth', None)


# Generating Book Recommendation for Famous Title
#Get 10 book recommendations for 'Macbeth' (by Shakespeare)
book_title = 'Macbeth'
Get_Recommendations(book_title, overall_similarity_mtrx, genre_sim_mtrx, alpha=0.7, top_n=10)


# Generating Book Recommendations from Random Titles
#Get recommendations for titles chosen at random
random_titles = df.sample(5)[['book_title','author']]

#get recommendations for the selected titles
for title,author in zip(random_titles.iloc[:,0],random_titles.iloc[:,1]):
    Get_Recommendations(title, overall_similarity_mtrx, genre_sim_mtrx, alpha=0.7, top_n=10)
    print('\n', 150*'_' + '\n')


# Generating Book Recommendations from User Input (titles only)
#Defining custom function that requests a book title from the user and returns relevant book recommendations
def Get_Recommendations_fromUser(top_n=10):
    while True:
        book_title = input('\nEnter book title: ')     
        recommendations = Get_Recommendations(book_title, overall_similarity_mtrx, genre_sim_mtrx, alpha=0.7, top_n=top_n)
        print('\n', 150*'_' + '\n', flush=True)
        if recommendations is not False:
            response = str(input('\n\nWould you like to get recommendations for more books? [Yes/no]\n')).lower().strip()
            if response in ['yes', 'y']: 
                continue 
            elif response in ['no', 'n']:
                print('\nThank you for trying the recommender.\nExiting...')
                break
            else: 
                print('\nResponse invalid.\nProcess terminating...')
                break


#Testing the function
#Execute the user recommender function
Get_Recommendations_fromUser()  # The Great Gatsby; Return of the king; Atomic Habit; a brief history of time; Critique of pure reason


# Generating Book Recommendations from User Query
#Define function to preprocess query from user 
def preprocess_query(query):
    #removing punctuations and whitespaces, and lowercasing
    query = ' '.join(re.findall(r'\b\w+\b', query.lower().strip()))
    #removing stop words 
    query = remove_stopwords(query, stopwords_multilang)
    #lemmatize query 
    query = lemmatize_text(query)
    #tokenize query 
    query = tokenizer.texts_to_sequences([query]) 
    #sequence padding 
    query = pad_sequences(query, maxlen=max_seq_len, padding='post', truncating='post')
    #return preprocessed query
    return query 

#Define recommendation function to recommend books from user query
def Get_Recommendations_forQuery(query=None, top_n=10):
    if query is None:
        query = str(input('\Enter book description: '))

    #Preprocess user's query
    query_processed = preprocess_query(query)
    
    #Encode the user query    
    query_embedding = embeddings_model.predict(query_processed)
    
    #Compute similarity with all book embeddings
    overall_sim_mtrx = cosine_similarity(query_embedding, book_embeddings).flatten()
    query_genre = [idx2word[word_token] if (word_token!=0 and idx2word[word_token].capitalize() in genres_df.columns) else None for word_token in query_processed[0]]
    query_genre = [q.capitalize() for q in query_genre if q is not None]
    if len(query_genre) > 0:
        book_indices = [i for i in df.index if set(query_genre).intersection(set(df['genres'].iloc[i].split(', ')))]
    else:
        #get scores by indices
        book_indices = [i for i in range(len(overall_sim_mtrx))]
    
    #Filter books to include books written in the same language as the target book
    book_indices_final = [i for i in book_indices if df['language'].iloc[i] == detect(query)]
    if not book_indices_final:
      book_indices_final = book_indices  

    #Create similarity scores for the filtered indices
    similarity_scores = [(i, overall_sim_mtrx[i]) for i in book_indices_final]

    #sort indices by cosine score and get top 10
    top_similarity_indices = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:top_n]
    top_similarity_indices = [idx for idx,score in top_similarity_indices]  #get indices only

    #Prepare DataFrame with recommended books and their details
    recommended_books = df.iloc[top_similarity_indices][['book_title', 'author', 'genres', 'cover_image_uri']] 
    recommended_books['Recommendation'] = recommended_books.apply(lambda row: f"{row['book_title']} (by {row['author']})  -  Genre: {row['genres'].split(', ')[0]}", axis=1)
    recommended_books.reset_index(drop=True, inplace=True)

    #Return book recommendations
    print(f"\nTop {int(top_n)} book recommendations:", flush=True)
    print(recommended_books['Recommendation'].to_frame().rename(lambda x:x+1))
    print('\n', flush=True)
    get_covers(recommended_books)
    return 


#Testing the function
#Queries to test out 
mystery_thriller_query = "Recommend a detective noir set in a corrupt little town, with a gritty detective and a femme fatale."
fantasy_adventure_query = "Recommend a high-fantasy epic about a quest to save the world from an ancient evil, filled with magic and mythical creatures."
philosophy_query = "Recommend a philosophy book about the nature of consciousness, reason and the objective-subjective distinction."

queries = [mystery_thriller_query, fantasy_adventure_query, philosophy_query] 
query_type = ['Mystery-Thriller', 'Fantasy-Adventure', 'Philosophy']

for query,query_type in zip(queries, query_type):
    print(f'\nRecommendations for {query_type} query:')
    Get_Recommendations_forQuery(query)
