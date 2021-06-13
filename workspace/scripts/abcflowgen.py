from typing import Tuple
import math


class Vec3:
    def __init__(self) -> None:
        self.x = 0
        self.y = 0
        self.z = 0

    def prod(self):
        return self.x * self.y * self.z


class Bound:
    def __ini__(self):
        self.min = Vec3()
        self.max = Vec3()


def abcflow(width,height,depth):
	min_value = 0.0031238
	max_value = 3.4641
	A = math.sqrt(3)
	B = math.sqrt(2)
	C = 1
	data = [None] * width * height * depth
	twopi = math.pi * 2
	sqrt6 = 2*math.sqrt(6)
	for z in range(0, depth):
		Z = twopi * z* 1.0/depth
		for y in range(0,height):
			Y = twopi * y* 1.0/height
			for x in range(0, width):
				ind = x + width * (y + z * height)
				X = twopi * x* 1.0/width
				val = math.sqrt(6 + 2 * A * math.sin(Z) * math.cos(Y) + 2*B*math.sin(Y) * math.cos(X) + sqrt6*math.sin(X) * math.cos(Z))
				val = int((val - min_value) / (max_value - min_value)  * 255 + 0.5)
				data[ind] = val
	return bytearray(data)


if __name__ == '__main__':
	data = abcflow(256,256,256)
	with open("../data/abcflow.raw",'wb') as f:
		f.write(data)

	with open("../data/abcflow.vifo",'w') as f:
		f.writelines("1\n")
		f.writelines("uchar\n")
		f.writelines("256 256 256\n")
		f.writelines("1 1 1\n")
		f.writelines("abcflow.raw")