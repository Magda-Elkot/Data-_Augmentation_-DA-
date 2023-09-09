import random
from random import shuffle
random.seed(1)

#stop words list
'''
The list of stop words contains commonly used words in the English language 
that are generally considered to be uninformative
'''
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
			'ours', 'ourselves', 'you', 'your', 'yours', 
			'yourself', 'yourselves', 'he', 'him', 'his', 
			'himself', 'she', 'her', 'hers', 'herself', 
			'it', 'its', 'itself', 'they', 'them', 'their', 
			'theirs', 'themselves', 'what', 'which', 'who', 
			'whom', 'this', 'that', 'these', 'those', 'am', 
			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at', 
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after', 
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again', 
			'further', 'then', 'once', 'here', 'there', 'when', 
			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
			'very', 's', 't', 'can', 'will', 'just', 'don', 
			'should', 'now', '']

#cleaning up text
'''
The get_only_chars function takes a string as input and performs several
operations to clean up the text. 

It removes apostrophes, replaces hyphens with spaces, converts the text to
lowercase, removes all non-alphabetic characters except spaces, and removes
extra spaces.
'''
import re
def get_only_chars(line):

    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:] # avoid lines starting with spaces at the beginning
    return clean_line

########################################################################
# Synonym replacement
# Replace n words in the sentence with synonyms from wordnet
########################################################################

#for the first time you use wordnet
'''
WordNet is a lexical database for the English language that provides information
about word meanings, relationships between words, and other linguistic information.
'''
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet 

'''
The synonym_replacement function takes two arguments: words, which is a list of words,
and n, which is the number of words to be replaced with synonyms.

The function first makes a copy of the input list using the copy method.

It then creates a list of unique words in the input list that are not in the 
stop words list.

The random.shuffle method is used to shuffle the list of words in random order.

The function then iterates over each word in the shuffled list and tries to replace
it with a random synonym using the get_synonyms function.

If a synonym is found, the word is replaced with a random synonym in the copied list
of words and the num_replaced counter is incremented.

The function continues to replace words until num_replaced equals n.

Finally, the function joins the list of words into a string and splits it back into
a list of words before returning.
'''
def synonym_replacement(words, n):
	new_words = words.copy()
	random_word_list = list(set([word for word in words if word not in stop_words]))
	random.shuffle(random_word_list)
	num_replaced = 0
	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)
		if len(synonyms) >= 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			#print("replaced", random_word, "with", synonym)
			num_replaced += 1
		if num_replaced >= n: #only replace up to n words
			break

	#this is stupid but we need it, trust me
	sentence = ' '.join(new_words)
	new_words = sentence.split(' ')

	return new_words

'''
The get_synonyms function takes a word as input and returns a list of synonyms for
that word.

It uses the wordnet.synsets method from the wordnet module to retrieve all
synsets (sets of synonyms) for the input word.

It then iterates over each synset and retrieves the lemma names (synonyms) for that synset.

The synonym is cleaned up by removing underscores, hyphens, and non-alphabetic characters,
and is added to a set of unique synonyms.

If the input word is in the set of synonyms, it is removed.

Finally, the function returns a list of unique synonyms for the input word.
'''
def get_synonyms(word):
	synonyms = set()
	for syn in wordnet.synsets(word): 
		for l in syn.lemmas(): 
			synonym = l.name().replace("_", " ").replace("-", " ").lower()
			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
			synonyms.add(synonym) 
	if word in synonyms:
		synonyms.remove(word)
	return list(synonyms)

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

'''
The random_deletion function takes two arguments: words, which is a list of words, and p,
which is the probability of deleting each word. For each word in the input list, the
function generates a random number between 0 and 1 using the random.uniform method.

If the random number is greater than p, the word is added to a new list of words called 
new_words.

If the random number is less than or equal to p, the word is deleted.

The function continues to iterate over the input list until all words have been processed.

If all words in the input list are deleted, the function chooses a random word from the
original list using the random.randint method and returns it in a list.

If no words are deleted, the function returns the original list of words.
'''

def random_deletion(words, p):

	#obviously, if there's only one word, don't delete it
	if len(words) == 1:
		return words

	#randomly delete words with probability p
	new_words = []
	for word in words:
		r = random.uniform(0, 1)
		if r > p:
			new_words.append(word)

	#if you end up deleting all words, just return a random word
	if len(new_words) == 0:
		rand_int = random.randint(0, len(words)-1)
		return [words[rand_int]]

	return new_words

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################
'''
The random_swap function takes two arguments: words, which is a list of words, and n,
which is the number of times to perform word swapping.

The function first makes a copy of the input list using the copy method.

It then repeatedly calls the "swap_word" function n times to perform random word swapping.

Finally, the function returns the modified list of words.
'''
def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

'''
The swap_word function takes a list of words as input and returns a new list of words
with two randomly selected words swapped.

The function first generates two random indices between 0 and the length of the input
list minus 1 using the random.randint method.

It then checks if the two indices are the same, and if so, generates new indices up to
three times until they are not the same.

If the function is unable to generate two different indices after three attempts,
it returns the original list of words without modification.

Otherwise, it swaps the words at the two indices using Python's list indexing and
assignment syntax.

Finally, the function returns the modified list of words with the two words swapped.
'''
def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################
'''
The random_insertion function takes two arguments: words, which is a list of words,
and n, which is the number of times to perform word insertion.

The function first makes a copy of the input list using the copy method.

It then repeatedly calls the add_word function n times to perform random word insertion.

Finally, the function returns the modified list of words.
'''
def random_insertion(words, n):
	new_words = words.copy()
	for _ in range(n):
		add_word(new_words)
	return new_words

'''
The add_word function takes a list of words as input and inserts a synonym of a randomly
selected word from the list at a random position in the list.

The function first selects a random word from the input list using the random.randint method.

It then calls the get_synonyms function to retrieve a list of synonyms for the selected word.

If no synonyms are found after 10 attempts, the function returns without modifying the 
input list.

Otherwise, it selects the first synonym from the list and generates a random index between
0 and the length of the input list using random.randint.

It then inserts the synonym at the selected index using the insert method of the list data type.

Finally, the function returns the modified list of words.
'''
def add_word(new_words):
	synonyms = []
	counter = 0
	while len(synonyms) < 1:
		random_word = new_words[random.randint(0, len(new_words)-1)]
		synonyms = get_synonyms(random_word)
		counter += 1
		if counter >= 10:
			return
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words)-1)
	new_words.insert(random_idx, random_synonym)

########################################################################
# main data augmentation function
########################################################################

'''
The eda function takes several arguments:

sentence: the input sentence to be augmented.

alpha_sr, alpha_ri, alpha_rs, and p_rd: the hyperparameters that control
the strength of each data augmentation technique.These parameters determine
the probability that each technique will be applied to a given word in
the input sentence (could be adjusted), alpha is nedded fot the equation
n = (alpha*l), Since long sentences have more words than short ones, they
can absorb more noise while maintaining their original class label. To
compensate, we vary the number of words changed (n)

num_aug: the number of augmented sentences to be generated (could be adjusted).
'''
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):
	'''
    The function first cleans the input sentence using the get_only_chars function,
    which removes any non-alphabetic characters.
    
    It then splits the sentence into a list of words and removes any empty strings
    from the list.

    The function then generates num_new_per_technique new sentences using each of
    the four data augmentation techniques. 
    
    The number of new sentences to generate for each technique is calculated as 
    int(num_aug/4)+1, which ensures that an equal number of new sentences are 
    generated for each technique.
    '''
	sentence = get_only_chars(sentence) #clean sentence
	words = sentence.split(' ') # split sentence into tokens
	words = [word for word in words if word != ''] #convert tokens from a string into a list
	num_words = len(words) # num_words: l ======> n = alpha*l
	
	augmented_sentences = []
	num_new_per_technique = int(num_aug/4)+1
    
    
    #For each technique, the function calls the corresponding augmentation function
    #(synonym_replacement, random_insertion, random_swap, or random_deletion) with
    #the appropriate hyperparameter values.
    
    #The resulting new sentences are appended to a list called augmented_sentences.

	#sr
	if (alpha_sr > 0):
        #alpha_sr: adjusted hyperparameter indicate the strength of sr technique
        #num_words(n): length of each sentence in the training set
        #n_sr: indicates num of random words to be replaced with their synonyms
		n_sr = max(1, int(alpha_sr*num_words))
		for _ in range(num_new_per_technique):
			a_words = synonym_replacement(words, n_sr)
			augmented_sentences.append(' '.join(a_words))

	#ri
	if (alpha_ri > 0):
        #alpha_ri: adjusted hyperparameter indicate the strength of ri technique
        #num_words(n): length of each sentence in the training set
        #n_ri: indicates num of times of random word insertions
		n_ri = max(1, int(alpha_ri*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_insertion(words, n_ri)
			augmented_sentences.append(' '.join(a_words))

	#rs
	if (alpha_rs > 0):
        #alpha_rs: adjusted hyperparameter indicate the strength of rs technique
        #num_words(n): length of each sentence in the training set
        #n_rs: indicates num of times of word swappings
		n_rs = max(1, int(alpha_rs*num_words))
		for _ in range(num_new_per_technique):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	#rd
	if (p_rd > 0):
        #p_rd is the adjusted parameter for rd which represents the probability
        #of the deletion of each word in the sentence
		for _ in range(num_new_per_technique):
			a_words = random_deletion(words, p_rd)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
	#shuffle(augmented_sentences)

    #ensures that the output of the eda function contains the desired number 
    #of augmented sentences, with a random subset of the possible augmentations
    #applied to the input sentence. The use of shuffling and random selection
    #helps to increase the variability and diversity of the output
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
	augmented_sentences.append(sentence)

	return augmented_sentences