# -*-coding:utf-8-*-

f = open('train.txt', 'w')
f.write('__label__1 i love you\n')
f.write('__label__1 he loves me\n')
f.write('__label__1 she likes baseball\n')
f.write('__label__0 i hate you\n')
f.write('__label__0 sorry for that\n')
f.write('__label__0 this is awful\n')
f.close()

f = open('test.txt', 'w')
f.write('sorry hate you')
f.close()

