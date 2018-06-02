
# The following code is from Baidu ba-dls-deepspeech: https://github.com/baidu-research/ba-dls-deepspeech/
# Which is under the Apache License:
#    Copyright 2015-2016 Baidu USA LLC. All rights reserved.

#    Apache License
#    Version 2.0, January 2004
#    http://www.apache.org/licenses/


char_map_str = """
<SPACE> 0
a 1
b 2
c 3
d 4
e 5
f 6
g 7
h 8
i 9
j 10
k 11
l 12
m 13
n 14
o 15
p 16
q 17
r 18
s 19
t 20
u 21
v 22
w 23
x 24
y 25
z 26
' 27

"""

char_map = {}
index_map = {}

for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch

index_map[0] = ' '

