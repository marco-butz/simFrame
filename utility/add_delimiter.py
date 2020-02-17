__author__= "Marco Butz"

import sys

with open(sys.argv[1], 'r') as myfile:
  data = myfile.read()

delimited = []
for i in data:
    if i != '\n':
        delimited.append(i)
        delimited.append(',')
    else:
        del delimited[-1]
        delimited.append('\n')

with open(sys.argv[1].split('.')[0] + '_delimited.txt', 'w+') as outFile:
    outFile.write("".join(delimited))
