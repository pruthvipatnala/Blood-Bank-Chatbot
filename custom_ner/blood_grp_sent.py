
b_grp = ['A+','A-','AB+','AB-','B+','B-','O+','O-']
end_sents = ['My blood group is ','My group is ','']

mid_sents = ['I have - blood']

bgrp_train_sentences = []

'''
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]
'''


for i in end_sents:
    for j in b_grp:
        bgrp_train_sentences.append('("'+i+j+'",{"entities":[('+str(len(i))+','+str(len(j+i))+',"BGRP"'+')]})')

for i in b_grp:
    s = mid_sents[0]
    a = s.index('-')
    b = a+len(i)
    k = s.replace('-',i)
    bgrp_train_sentences.append('("'+i+j+'",{"entities":[('+str(a)+','+str(b)+',"BGRP"'+')]})')
    


print(','.join(bgrp_train_sentences))












