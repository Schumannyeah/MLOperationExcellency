# modules we'll use
import pandas as pd
import numpy as np

# helpful character encoding module
import charset_normalizer

# set seed for reproducibility
np.random.seed(0)

# start with a string
before = "This is the euro symbol: €"

# check to see what datatype it is
print(type(before))

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("utf-8", errors="replace")

# check the type
print(type(after))

print(after)



# start with a string
before = "This is the euro symbol: €"

# encode it to a different encoding, replacing characters that raise errors
after = before.encode("ascii", errors = "replace")

# convert it back to utf-8
print(after.decode("ascii"))

# We've lost the original underlying byte string! It's been
# replaced with the underlying byte string for the unknown character :(




# look at the first ten thousand bytes to guess the character encoding
with open("../dataset/ks-projects-201801.csv", 'rb') as rawdata:
    result = charset_normalizer.detect(rawdata.read(10000))

# check what the character encoding might be
print(result)




# read in the file with the encoding detected by charset_normalizer
kickstarter_2016 = pd.read_csv("../dataset/ks-projects-201612.csv", encoding='Windows-1252')

# look at the first few lines
print(kickstarter_2016.head())



# save our file (will be saved as UTF-8 by default!)
kickstarter_2016.to_csv("ks-projects-201612-utf8.csv")
