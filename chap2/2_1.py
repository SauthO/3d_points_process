import sys
import struct


#print(sys.argv[0])
#print(sys.argv[1])

with open(sys.argv[1], "rb") as f:
    flg = True
    while True:
        line = f.readline()
        print(line)
        if b"end_header" in line:
            break
        if b"vertex" in line and flg:
            print(line.split(b" ")[-1])
            vnum = int(line.split(b" ")[-1]) # num of vertices
            flg = False
        if b"face" in line:
            fnum = int(line.split(b" ")[-1]) # num of faces
    
    # read vertex
    for i in range(vnum):
        for j in range(3):
            print(struct.unpack("f", f.read(4))[0], end="")
        print("")
    
    # read face
    for i in range(fnum):
        n = struct.unpack("B", f.read(1))[0]
        #print(f"n:{n}")
        for j in range(n):
            print(struct.unpack("i", f.read(4))[0], end="")
        print("")