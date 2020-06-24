"""
Last amended: 6th June, 2020
My folder: C:\Users\ashok\OneDrive\Documents\sentiment_analysis
Virtual Machine: lubuntu_machinelearning_I

Objective:
    Text clustering of wiki documents

About our text files:
    Our files are on following subjects:
            1. Quantum Mechanics
            2. Religion
            3. Legal
            4. Psychology
    Total text files 12

TODO
    HASHVECTORIZER

"""

###################### 1. Call libraries #####################
# 1.0 Clear memory
%reset -f
# 1.1 Array and data-manipulation libraries
import numpy as np
import pandas as pd

# 1.2 sklearn modeling libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 1.3 Text processing module
import re

# 1.4 Miscellenous
import os

# 2.0 Cleaning of text using regular expressions:

## 2.1 Regular Expression usage through re module:
#      Raw string notations are generally used:
#       '\n' is a single char--newline
#       r'\n' a 2-char string of \ and n

# 2.2 Replace bracketed numbers with space
x = "[8]OK good[6] [6] [5]done"
result= re.sub(r'[\[0-9\]]',' ', x)
result

# 2.3 Remove newlines
x = "OK \n good\n  \ndone"
result= re.sub('\n',' ', x)
result= re.sub(r'\n',' ', x)
result

# 2.5 Remove apostrophe
x= "Planck's solution"
x="After that it's just a matter "
result= re.sub('\'s',' ', x)      # Either this
result= re.sub('[\'s]',' ', x)    # Or this  
result

# 2.6 Remove html tags
#     https://stackoverflow.com/a/12982689/3282777
#     https://stackoverflow.com/a/3075150/3282777
x = " <title>Cultural universal</title>      <ns> </ns>      <id>       </id>      <revision>        <id>         </id>        <parentid>         </parentid>        <timestamp>    -  -  T  :  :  Z</timestamp>"

# 2.6.1 Compiling creates a pattern object
#       A pattern object also has its own methods/attribtes
clean_greedy = re.compile('<.*>') # This object is greedy
clean = re.compile('<.*?>')       # This object is not greedy
re.sub(clean, "", x)
re.sub(clean_greedy, "", x)

# 3.0 Read files as text streams and also clean them
#     https://docs.python.org/3/library/io.html#text-i-o

pathToclusteringFiles="C:\\Users\\ashok\\OneDrive\\Documents\\sentiment_analysis\\textclustering"
pathToclusteringFiles="D:\\data\\OneDrive\\Documents\\sentiment_analysis\\textclustering"

os.chdir(pathToclusteringFiles)
os.listdir()
len(os.listdir())   # 12 txt files

# 3.1 Experiment: Understanding a text-stream
#     Python creates a text-stream when reading
#     a text file. That is, rather than reading
#     a file in one go, one can read line-by-line
#     or a specific number of chars at a time

filelist = os.listdir()

# 3.1.1 Create a text stream or an iterator that outputs text
#       on demand
text_stream = open(filelist[0], "r",  encoding="utf8")
type(text_stream)      # TextIOWrapper

# 3.1.2 Some attributes/methods
text_stream.encoding
text_stream.read(10)   # Read at most 'size' characters
                       #   from stream as a single str
                       #    If size = -1, all chars are read
text_stream.read(10)   # Read next 'size' characters

# 3.1.3 It also behaves as iterable
text_stream = open(filelist[0], "r",  encoding="utf8")
t = text_stream.__iter__()
next(t)

# 3.1.4 Read few lines
text_stream = open(filelist[0], "r",  encoding="utf8")
text_stream.readlines(1)   # Read until newline or EOF and return
                           # a single str within a list
text_stream.readlines(1)   # Read next line or if nothing, return empty string

# 3.1.5 Read all lines as a list
text_stream = open(filelist[0], "r",  encoding="utf8")
type(out)    # list
len(out)     # 3
out = text_stream.readlines()
out

# 3.2
lines = []
for i in os.listdir():
    # 3.2.1 Create text-stream
    text_file = open(i, "r",  encoding="utf8")
    # 3.2.2 Read complete file in a list of strings
    tx = text_file.readlines()
    # 3.2.3 Join all strings in the list
    tx = " ".join(tx)
    # 3.2.4 Create pattern object to remove html tags
    clean = re.compile('<.*?>')  #
    tx = re.sub(clean, '', tx)

    # 3.2.5 Replace bracketed numbers with space
    tx= re.sub(r'[\[0-9\]]',' ', tx)

    # 3.2.6
    tx= re.sub('\n',' ', tx)    # Remove newlines
    tx= re.sub('\'s',' ', tx)   # Remove apostrophes
    tx= re.sub('\'s',' ', tx)

    # 3.2.7 Remove URLs. In MULTILINE mode also matches immediately
    #       after each newline.
    tx = re.sub(r'^https?:\/\/.*[\r\n]*', '', tx, flags=re.MULTILINE)

    # 3.2.8 Other Miscellenous
    tx = re.sub('[*|\(\)\{\}]', " ",tx)
    tx = re.sub('[=]*', "",tx)

    # 3.2.9 Tags may take such forms also
    #       < == &lt;   > == &gt;
    clean = re.compile('&lt;')
    tx = re.sub(clean, '', tx)
    clean = re.compile('&gt;')
    tx = re.sub(clean, '', tx)
    clean = re.compile('&quot;')
    tx = re.sub(clean, '', tx)

    # 3.2.10 Finally append  this file
    #        to our list as a cleaned string
    lines.append(tx)

# 3.3 So what do we have
type(lines)     # list
lines
len(lines)      # 12; same as number of files

# 3.5 Clustering text documents
## One Way: Just use 'tf' and not 'idf'

#  3.5.1 Convert a collection of text documents to a matrix
#        of token counts. This implementation produces a
#        sparse representation of the counts using
#        scipy.sparse.csr_matrix.
#        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
vec = CountVectorizer()
matrix = vec.fit_transform(lines)
matrix
matrix.shape   # (12, 1466)

# 3.5.2 Let us see this sparse matrix in a dataframe
#       We have token-counts
pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

## 3.6 Better way
#      Use both tf and idf
#      https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vec = TfidfVectorizer(use_idf=True,  stop_words='english')
matrix = vec.fit_transform(lines)
matrix.shape   # (12, 1311)  # Stop words have been removed

# 3.6.1 Have a look at our tf-idf values
pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())

# 3.7 Finally start clustering
number_of_clusters=  4   # As many as types of documents

# 3.71. Instantiate KMeans object
km = KMeans(
            n_clusters=number_of_clusters,
            max_iter=500
            )

# 3.72 Train our model
km.fit(matrix)

# 3.7.3
km.labels_

# 3.7.4
km.inertia_   # 6.731

re.split(r'_','l1_q.txt')

# 4.0 Let us arrange our results
#     Put filenames and cluster-labels at one place

# 4.1 First, we modify filenames. Example:
re.split(r'_','l1_q.txt')    # Just see what happens

# 4.2
results = pd.DataFrame()
modified_filenames = []
for i in os.listdir():
    modified_filenames.append(re.split(r'_',i)[1])

# 4.2.1
modified_filenames

# 4.2.2
results['text'] = modified_filenames
results['category'] = km.labels_
results

# 4.2.3 Sort on 'text' column to
#       clearly see match of cluster
#       labels with filenames
results.sort_values('text')

############################################################

"""
Refer: https://docs.python.org/3/library/re.html
       https://docs.python.org/3/howto/regex.html#regex-howto
Regular expressions can contain both special and
ordinary characters. Most ordinary characters,
like 'A', 'a', or '0', are the simplest regular
expressions; they simply match themselves. You can
concatenate ordinary characters, so last matches the
string 'last'.

.  (Dot.) Matches any character except a newline.
^  (Caret.) Matches start of the string
$  Matches end of the string
*  Causes the resulting RE to match 0 or more repetitions.
    ab* will match ‘a’, ‘ab’, or ‘a’ followed by any number of ‘b’s.
+  Causes the resulting RE to match 1 or more repetitions of the preceding RE.
   ab+ will match ‘a’ followed by any non-zero number of ‘b’s
?  Causes the resulting RE to match 0 or 1 repetitions of the preceding RE.
[] Used to indicate a set of characters. In a set:
   -Characters can be listed individually, e.g. [amk] will match 'a', 'm', or 'k'.
   -Ranges of characters can be indicated by giving two characters and separating
    them by a '-', for example [a-z] will match any lowercase ASCII letter,
    [0-5][0-9] will match all the two-digits numbers from 00 to 59,
    and [0-9A-Fa-f] will match any hexadecimal digit.
   -Special characters lose their special meaning inside sets.
    For example, [(+*)] will match any of the literal characters
    '(', '+', '*', or ')'.
   -Character classes such as \w or \S are also accepted inside a set,
   -Characters that are not within a range can be matched by complementing
    the set. If the first character of the set is '^', all the characters
    that are not in the set will be matched. For example, [^5] will match
    any character except '5'. ^ has no special meaning if it’s not the first
    character in the set.
   -To match a literal ']' inside a set, precede it with a backslash, or place
    it at the beginning of the set.
{m,n}
    Causes resulting RE to match from m to n repetitions of the preceding RE,
    attempting to match as many repetitions as possible. For example, a{3,5}
    will match from 3 to 5 'a' characters. Omitting m specifies a lower bound
    of zero, and omitting n specifies an infinite upper bound.
"""

"""
Greedy vs non-greedy
a.  Let your string input be: 101000000000100.
b.  Using 1.*1, * is greedy - it will match all the way to the end,
    and then backtrack until it can match 1, leaving you with 1010000000001.
c.  .*? is non-greedy. * will match nothing, but then will try to match extra
    characters until it matches 1, eventually matching 101.
"""
