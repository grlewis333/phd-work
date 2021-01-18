import numba

@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g1_0(f,Df):
	for i0 in numba.prange(0, f.shape[0]-1):
		Df[i0, 0] = f[i0+1] - f[i0]

	i0 = f.shape[0]-1
	Df[i0, 0] = 0


print("1/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _gt1_0(Df,f):
	i0 = 0
	f[i0] = -Df[i0, 0]

	for i0 in numba.prange(1, Df.shape[0]-1):
		f[i0] = Df[i0-1, 0] - Df[i0, 0]

	i0 = Df.shape[0]-1
	f[i0] = Df[i0-1, 0]


print("2/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g21_0(f,Df):
	i0 = 0
	Df[i0, 0] = f[i0]

	for i0 in numba.prange(1, f.shape[0]):
		Df[i0, 0] = f[i0] - f[i0-1]


print("3/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2t1_0(Df,f):
	for i0 in numba.prange(0, Df.shape[0]-1):
		f[i0] = Df[i0, 0] - Df[i0+1, 0]

	i0 = Df.shape[0]-1
	f[i0] = Df[i0, 0]


print("4/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g1_1(f,Df):
	for i0 in numba.prange(0, f.shape[0]-1):
		for i1 in range(f.shape[1]):
			Df[i0,i1, 0] = f[i0+1,i1] - f[i0,i1]

	i0 = f.shape[0]-1
	for i1 in range(f.shape[1]):
		Df[i0,i1, 0] = 0


print("5/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _gt1_1(Df,f):
	i0 = 0
	for i1 in range(Df.shape[1]):
		f[i0,i1] = -Df[i0,i1, 0]

	for i0 in numba.prange(1, Df.shape[0]-1):
		for i1 in range(Df.shape[1]):
			f[i0,i1] = Df[i0-1,i1, 0] - Df[i0,i1, 0]

	i0 = Df.shape[0]-1
	for i1 in range(Df.shape[1]):
		f[i0,i1] = Df[i0-1,i1, 0]


print("6/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g21_1(f,Df):
	i0 = 0
	for i1 in range(f.shape[1]):
		Df[i0,i1, 0] = f[i0,i1]

	for i0 in numba.prange(1, f.shape[0]):
		for i1 in range(f.shape[1]):
			Df[i0,i1, 0] = f[i0,i1] - f[i0-1,i1]


print("7/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2t1_1(Df,f):
	for i0 in numba.prange(0, Df.shape[0]-1):
		for i1 in range(Df.shape[1]):
			f[i0,i1] = Df[i0,i1, 0] - Df[i0+1,i1, 0]

	i0 = Df.shape[0]-1
	for i1 in range(Df.shape[1]):
		f[i0,i1] = Df[i0,i1, 0]


print("8/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2_0(f,Df):
	for i0 in numba.prange(0, f.shape[0]-1):
		for i1 in range(0, f.shape[1]-1):
			Df[i0,i1, 0] = f[i0+1,i1] - f[i0,i1]
			Df[i0,i1, 1] = f[i0,i1+1] - f[i0,i1]

#		i0 is looping
		i1 = f.shape[1]-1
		Df[i0,i1, 0] = f[i0+1,i1] - f[i0,i1]
		Df[i0,i1, 1] = 0

	i0 = f.shape[0]-1
	for i1 in numba.prange(0, f.shape[1]-1):
		Df[i0,i1, 0] = 0
		Df[i0,i1, 1] = f[i0,i1+1] - f[i0,i1]

#	i0 = f.shape[0]-1
	i1 = f.shape[1]-1
	Df[i0,i1, 0] = 0
	Df[i0,i1, 1] = 0


print("9/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _gt2_0(Df,f):
	i0 = 0
	i1 = 0
	f[i0,i1] = -Df[i0,i1, 0]
	f[i0,i1] -= Df[i0,i1, 1]

#	i0 = 0
	for i1 in numba.prange(1, Df.shape[1]-1):
		f[i0,i1] = -Df[i0,i1, 0]
		f[i0,i1] += Df[i0,i1-1, 1] - Df[i0,i1, 1]

#	i0 = 0
	i1 = Df.shape[1]-1
	f[i0,i1] = -Df[i0,i1, 0]
	f[i0,i1] += Df[i0,i1-1, 1 ]

	for i0 in numba.prange(1, Df.shape[0]-1):
		i1 = 0
		f[i0,i1] = Df[i0-1,i1, 0] - Df[i0,i1, 0]
		f[i0,i1] -= Df[i0,i1, 1]

#		i0 is looping
		for i1 in range(1, Df.shape[1]-1):
			f[i0,i1] = Df[i0-1,i1, 0] - Df[i0,i1, 0]
			f[i0,i1] += Df[i0,i1-1, 1] - Df[i0,i1, 1]

#		i0 is looping
		i1 = Df.shape[1]-1
		f[i0,i1] = Df[i0-1,i1, 0] - Df[i0,i1, 0]
		f[i0,i1] += Df[i0,i1-1, 1 ]

	i0 = Df.shape[0]-1
	i1 = 0
	f[i0,i1] = Df[i0-1,i1, 0]
	f[i0,i1] -= Df[i0,i1, 1]

#	i0 = Df.shape[0]-1
	for i1 in numba.prange(1, Df.shape[1]-1):
		f[i0,i1] = Df[i0-1,i1, 0]
		f[i0,i1] += Df[i0,i1-1, 1] - Df[i0,i1, 1]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	f[i0,i1] = Df[i0-1,i1, 0]
	f[i0,i1] += Df[i0,i1-1, 1 ]


print("10/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g22_0(f,Df):
	i0 = 0
	i1 = 0
	Df[i0,i1, 0] = f[i0,i1]
	Df[i0,i1, 1] = f[i0,i1]

#	i0 = 0
	for i1 in numba.prange(1, f.shape[1]):
		Df[i0,i1, 0] = f[i0,i1]
		Df[i0,i1, 1] = f[i0,i1] - f[i0,i1-1]

	for i0 in numba.prange(1, f.shape[0]):
		i1 = 0
		Df[i0,i1, 0] = f[i0,i1] - f[i0-1,i1]
		Df[i0,i1, 1] = f[i0,i1]

#		i0 is looping
		for i1 in range(1, f.shape[1]):
			Df[i0,i1, 0] = f[i0,i1] - f[i0-1,i1]
			Df[i0,i1, 1] = f[i0,i1] - f[i0,i1-1]


print("11/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2t2_0(Df,f):
	for i0 in numba.prange(0, Df.shape[0]-1):
		for i1 in range(0, Df.shape[1]-1):
			f[i0,i1] = Df[i0,i1, 0] - Df[i0+1,i1, 0]
			f[i0,i1] += Df[i0,i1, 1] - Df[i0,i1+1, 1]

#		i0 is looping
		i1 = Df.shape[1]-1
		f[i0,i1] = Df[i0,i1, 0] - Df[i0+1,i1, 0]
		f[i0,i1] += Df[i0,i1, 1 ]

	i0 = Df.shape[0]-1
	for i1 in numba.prange(0, Df.shape[1]-1):
		f[i0,i1] = Df[i0,i1, 0]
		f[i0,i1] += Df[i0,i1, 1] - Df[i0,i1+1, 1]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	f[i0,i1] = Df[i0,i1, 0]
	f[i0,i1] += Df[i0,i1, 1 ]


print("12/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2_1(f,Df):
	for i0 in numba.prange(0, f.shape[0]-1):
		for i1 in range(0, f.shape[1]-1):
			for i2 in range(f.shape[2]):
				Df[i0,i1,i2, 0] = f[i0+1,i1,i2] - f[i0,i1,i2]
				Df[i0,i1,i2, 1] = f[i0,i1+1,i2] - f[i0,i1,i2]

#		i0 is looping
		i1 = f.shape[1]-1
		for i2 in range(f.shape[2]):
			Df[i0,i1,i2, 0] = f[i0+1,i1,i2] - f[i0,i1,i2]
			Df[i0,i1,i2, 1] = 0

	i0 = f.shape[0]-1
	for i1 in numba.prange(0, f.shape[1]-1):
		for i2 in range(f.shape[2]):
			Df[i0,i1,i2, 0] = 0
			Df[i0,i1,i2, 1] = f[i0,i1+1,i2] - f[i0,i1,i2]

#	i0 = f.shape[0]-1
	i1 = f.shape[1]-1
	for i2 in range(f.shape[2]):
		Df[i0,i1,i2, 0] = 0
		Df[i0,i1,i2, 1] = 0


print("13/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _gt2_1(Df,f):
	i0 = 0
	i1 = 0
	for i2 in range(Df.shape[2]):
		f[i0,i1,i2] = -Df[i0,i1,i2, 0]
		f[i0,i1,i2] -= Df[i0,i1,i2, 1]

#	i0 = 0
	for i1 in numba.prange(1, Df.shape[1]-1):
		for i2 in range(Df.shape[2]):
			f[i0,i1,i2] = -Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]

#	i0 = 0
	i1 = Df.shape[1]-1
	for i2 in range(Df.shape[2]):
		f[i0,i1,i2] = -Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]

	for i0 in numba.prange(1, Df.shape[0]-1):
		i1 = 0
		for i2 in range(Df.shape[2]):
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
			f[i0,i1,i2] -= Df[i0,i1,i2, 1]

#		i0 is looping
		for i1 in range(1, Df.shape[1]-1):
			for i2 in range(Df.shape[2]):
				f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
				f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]

#		i0 is looping
		i1 = Df.shape[1]-1
		for i2 in range(Df.shape[2]):
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]

	i0 = Df.shape[0]-1
	i1 = 0
	for i2 in range(Df.shape[2]):
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
		f[i0,i1,i2] -= Df[i0,i1,i2, 1]

#	i0 = Df.shape[0]-1
	for i1 in numba.prange(1, Df.shape[1]-1):
		for i2 in range(Df.shape[2]):
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	for i2 in range(Df.shape[2]):
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]


print("14/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g22_1(f,Df):
	i0 = 0
	i1 = 0
	for i2 in range(f.shape[2]):
		Df[i0,i1,i2, 0] = f[i0,i1,i2]
		Df[i0,i1,i2, 1] = f[i0,i1,i2]

#	i0 = 0
	for i1 in numba.prange(1, f.shape[1]):
		for i2 in range(f.shape[2]):
			Df[i0,i1,i2, 0] = f[i0,i1,i2]
			Df[i0,i1,i2, 1] = f[i0,i1,i2] - f[i0,i1-1,i2]

	for i0 in numba.prange(1, f.shape[0]):
		i1 = 0
		for i2 in range(f.shape[2]):
			Df[i0,i1,i2, 0] = f[i0,i1,i2] - f[i0-1,i1,i2]
			Df[i0,i1,i2, 1] = f[i0,i1,i2]

#		i0 is looping
		for i1 in range(1, f.shape[1]):
			for i2 in range(f.shape[2]):
				Df[i0,i1,i2, 0] = f[i0,i1,i2] - f[i0-1,i1,i2]
				Df[i0,i1,i2, 1] = f[i0,i1,i2] - f[i0,i1-1,i2]


print("15/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2t2_1(Df,f):
	for i0 in numba.prange(0, Df.shape[0]-1):
		for i1 in range(0, Df.shape[1]-1):
			for i2 in range(Df.shape[2]):
				f[i0,i1,i2] = Df[i0,i1,i2, 0] - Df[i0+1,i1,i2, 0]
				f[i0,i1,i2] += Df[i0,i1,i2, 1] - Df[i0,i1+1,i2, 1]

#		i0 is looping
		i1 = Df.shape[1]-1
		for i2 in range(Df.shape[2]):
			f[i0,i1,i2] = Df[i0,i1,i2, 0] - Df[i0+1,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1,i2, 1 ]

	i0 = Df.shape[0]-1
	for i1 in numba.prange(0, Df.shape[1]-1):
		for i2 in range(Df.shape[2]):
			f[i0,i1,i2] = Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1,i2, 1] - Df[i0,i1+1,i2, 1]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	for i2 in range(Df.shape[2]):
		f[i0,i1,i2] = Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1,i2, 1 ]


print("16/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g3_0(f,Df):
	for i0 in numba.prange(0, f.shape[0]-1):
		for i1 in range(0, f.shape[1]-1):
			for i2 in range(0, f.shape[2]-1):
				Df[i0,i1,i2, 0] = f[i0+1,i1,i2] - f[i0,i1,i2]
				Df[i0,i1,i2, 1] = f[i0,i1+1,i2] - f[i0,i1,i2]
				Df[i0,i1,i2, 2] = f[i0,i1,i2+1] - f[i0,i1,i2]

#			i0 is looping
#			i1 is looping
			i2 = f.shape[2]-1
			Df[i0,i1,i2, 0] = f[i0+1,i1,i2] - f[i0,i1,i2]
			Df[i0,i1,i2, 1] = f[i0,i1+1,i2] - f[i0,i1,i2]
			Df[i0,i1,i2, 2] = 0

#		i0 is looping
		i1 = f.shape[1]-1
		for i2 in range(0, f.shape[2]-1):
			Df[i0,i1,i2, 0] = f[i0+1,i1,i2] - f[i0,i1,i2]
			Df[i0,i1,i2, 1] = 0
			Df[i0,i1,i2, 2] = f[i0,i1,i2+1] - f[i0,i1,i2]

#		i0 is looping
#		i1 = f.shape[1]-1
		i2 = f.shape[2]-1
		Df[i0,i1,i2, 0] = f[i0+1,i1,i2] - f[i0,i1,i2]
		Df[i0,i1,i2, 1] = 0
		Df[i0,i1,i2, 2] = 0

	i0 = f.shape[0]-1
	for i1 in numba.prange(0, f.shape[1]-1):
		for i2 in range(0, f.shape[2]-1):
			Df[i0,i1,i2, 0] = 0
			Df[i0,i1,i2, 1] = f[i0,i1+1,i2] - f[i0,i1,i2]
			Df[i0,i1,i2, 2] = f[i0,i1,i2+1] - f[i0,i1,i2]

#		i0 = f.shape[0]-1
#		i1 is looping
		i2 = f.shape[2]-1
		Df[i0,i1,i2, 0] = 0
		Df[i0,i1,i2, 1] = f[i0,i1+1,i2] - f[i0,i1,i2]
		Df[i0,i1,i2, 2] = 0

#	i0 = f.shape[0]-1
	i1 = f.shape[1]-1
	for i2 in numba.prange(0, f.shape[2]-1):
		Df[i0,i1,i2, 0] = 0
		Df[i0,i1,i2, 1] = 0
		Df[i0,i1,i2, 2] = f[i0,i1,i2+1] - f[i0,i1,i2]

#	i0 = f.shape[0]-1
#	i1 = f.shape[1]-1
	i2 = f.shape[2]-1
	Df[i0,i1,i2, 0] = 0
	Df[i0,i1,i2, 1] = 0
	Df[i0,i1,i2, 2] = 0


print("17/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _gt3_0(Df,f):
	i0 = 0
	i1 = 0
	i2 = 0
	f[i0,i1,i2] = -Df[i0,i1,i2, 0]
	f[i0,i1,i2] -= Df[i0,i1,i2, 1]
	f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#	i0 = 0
#	i1 = 0
	for i2 in numba.prange(1, Df.shape[2]-1):
		f[i0,i1,i2] = -Df[i0,i1,i2, 0]
		f[i0,i1,i2] -= Df[i0,i1,i2, 1]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#	i0 = 0
#	i1 = 0
	i2 = Df.shape[2]-1
	f[i0,i1,i2] = -Df[i0,i1,i2, 0]
	f[i0,i1,i2] -= Df[i0,i1,i2, 1]
	f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

#	i0 = 0
	for i1 in numba.prange(1, Df.shape[1]-1):
		i2 = 0
		f[i0,i1,i2] = -Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
		f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#		i0 = 0
#		i1 is looping
		for i2 in range(1, Df.shape[2]-1):
			f[i0,i1,i2] = -Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
			f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#		i0 = 0
#		i1 is looping
		i2 = Df.shape[2]-1
		f[i0,i1,i2] = -Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

#	i0 = 0
	i1 = Df.shape[1]-1
	i2 = 0
	f[i0,i1,i2] = -Df[i0,i1,i2, 0]
	f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
	f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#	i0 = 0
#	i1 = Df.shape[1]-1
	for i2 in numba.prange(1, Df.shape[2]-1):
		f[i0,i1,i2] = -Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#	i0 = 0
#	i1 = Df.shape[1]-1
	i2 = Df.shape[2]-1
	f[i0,i1,i2] = -Df[i0,i1,i2, 0]
	f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
	f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

	for i0 in numba.prange(1, Df.shape[0]-1):
		i1 = 0
		i2 = 0
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
		f[i0,i1,i2] -= Df[i0,i1,i2, 1]
		f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#		i0 is looping
#		i1 = 0
		for i2 in range(1, Df.shape[2]-1):
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
			f[i0,i1,i2] -= Df[i0,i1,i2, 1]
			f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#		i0 is looping
#		i1 = 0
		i2 = Df.shape[2]-1
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
		f[i0,i1,i2] -= Df[i0,i1,i2, 1]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

#		i0 is looping
		for i1 in range(1, Df.shape[1]-1):
			i2 = 0
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
			f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#			i0 is looping
#			i1 is looping
			for i2 in range(1, Df.shape[2]-1):
				f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
				f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
				f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#			i0 is looping
#			i1 is looping
			i2 = Df.shape[2]-1
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
			f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

#		i0 is looping
		i1 = Df.shape[1]-1
		i2 = 0
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
		f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#		i0 is looping
#		i1 = Df.shape[1]-1
		for i2 in range(1, Df.shape[2]-1):
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
			f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#		i0 is looping
#		i1 = Df.shape[1]-1
		i2 = Df.shape[2]-1
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0] - Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

	i0 = Df.shape[0]-1
	i1 = 0
	i2 = 0
	f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
	f[i0,i1,i2] -= Df[i0,i1,i2, 1]
	f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#	i0 = Df.shape[0]-1
#	i1 = 0
	for i2 in numba.prange(1, Df.shape[2]-1):
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
		f[i0,i1,i2] -= Df[i0,i1,i2, 1]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#	i0 = Df.shape[0]-1
#	i1 = 0
	i2 = Df.shape[2]-1
	f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
	f[i0,i1,i2] -= Df[i0,i1,i2, 1]
	f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

#	i0 = Df.shape[0]-1
	for i1 in numba.prange(1, Df.shape[1]-1):
		i2 = 0
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
		f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#		i0 = Df.shape[0]-1
#		i1 is looping
		for i2 in range(1, Df.shape[2]-1):
			f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
			f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#		i0 = Df.shape[0]-1
#		i1 is looping
		i2 = Df.shape[2]-1
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1] - Df[i0,i1,i2, 1]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	i2 = 0
	f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
	f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
	f[i0,i1,i2] -= Df[i0,i1,i2, 2]

#	i0 = Df.shape[0]-1
#	i1 = Df.shape[1]-1
	for i2 in numba.prange(1, Df.shape[2]-1):
		f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
		f[i0,i1,i2] += Df[i0,i1,i2-1, 2] - Df[i0,i1,i2, 2]

#	i0 = Df.shape[0]-1
#	i1 = Df.shape[1]-1
	i2 = Df.shape[2]-1
	f[i0,i1,i2] = Df[i0-1,i1,i2, 0]
	f[i0,i1,i2] += Df[i0,i1-1,i2, 1 ]
	f[i0,i1,i2] += Df[i0,i1,i2-1, 2 ]


print("18/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g23_0(f,Df):
	i0 = 0
	i1 = 0
	i2 = 0
	Df[i0,i1,i2, 0] = f[i0,i1,i2]
	Df[i0,i1,i2, 1] = f[i0,i1,i2]
	Df[i0,i1,i2, 2] = f[i0,i1,i2]

#	i0 = 0
#	i1 = 0
	for i2 in numba.prange(1, f.shape[2]):
		Df[i0,i1,i2, 0] = f[i0,i1,i2]
		Df[i0,i1,i2, 1] = f[i0,i1,i2]
		Df[i0,i1,i2, 2] = f[i0,i1,i2] - f[i0,i1,i2-1]

#	i0 = 0
	for i1 in numba.prange(1, f.shape[1]):
		i2 = 0
		Df[i0,i1,i2, 0] = f[i0,i1,i2]
		Df[i0,i1,i2, 1] = f[i0,i1,i2] - f[i0,i1-1,i2]
		Df[i0,i1,i2, 2] = f[i0,i1,i2]

#		i0 = 0
#		i1 is looping
		for i2 in range(1, f.shape[2]):
			Df[i0,i1,i2, 0] = f[i0,i1,i2]
			Df[i0,i1,i2, 1] = f[i0,i1,i2] - f[i0,i1-1,i2]
			Df[i0,i1,i2, 2] = f[i0,i1,i2] - f[i0,i1,i2-1]

		for i0 in numba.prange(1, f.shape[0]):
			i1 = 0
			i2 = 0
			Df[i0,i1,i2, 0] = f[i0,i1,i2] - f[i0-1,i1,i2]
			Df[i0,i1,i2, 1] = f[i0,i1,i2]
			Df[i0,i1,i2, 2] = f[i0,i1,i2]

#			i0 is looping
#			i1 = 0
			for i2 in range(1, f.shape[2]):
				Df[i0,i1,i2, 0] = f[i0,i1,i2] - f[i0-1,i1,i2]
				Df[i0,i1,i2, 1] = f[i0,i1,i2]
				Df[i0,i1,i2, 2] = f[i0,i1,i2] - f[i0,i1,i2-1]

#			i0 is looping
			for i1 in range(1, f.shape[1]):
				i2 = 0
				Df[i0,i1,i2, 0] = f[i0,i1,i2] - f[i0-1,i1,i2]
				Df[i0,i1,i2, 1] = f[i0,i1,i2] - f[i0,i1-1,i2]
				Df[i0,i1,i2, 2] = f[i0,i1,i2]

#				i0 is looping
#				i1 is looping
				for i2 in range(1, f.shape[2]):
					Df[i0,i1,i2, 0] = f[i0,i1,i2] - f[i0-1,i1,i2]
					Df[i0,i1,i2, 1] = f[i0,i1,i2] - f[i0,i1-1,i2]
					Df[i0,i1,i2, 2] = f[i0,i1,i2] - f[i0,i1,i2-1]


print("19/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2t3_0(Df,f):
	for i0 in numba.prange(0, Df.shape[0]-1):
		for i1 in range(0, Df.shape[1]-1):
			for i2 in range(0, Df.shape[2]-1):
				f[i0,i1,i2] = Df[i0,i1,i2, 0] - Df[i0+1,i1,i2, 0]
				f[i0,i1,i2] += Df[i0,i1,i2, 1] - Df[i0,i1+1,i2, 1]
				f[i0,i1,i2] += Df[i0,i1,i2, 2] - Df[i0,i1,i2+1, 2]

#			i0 is looping
#			i1 is looping
			i2 = Df.shape[2]-1
			f[i0,i1,i2] = Df[i0,i1,i2, 0] - Df[i0+1,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1,i2, 1] - Df[i0,i1+1,i2, 1]
			f[i0,i1,i2] += Df[i0,i1,i2, 2 ]

#		i0 is looping
		i1 = Df.shape[1]-1
		for i2 in range(0, Df.shape[2]-1):
			f[i0,i1,i2] = Df[i0,i1,i2, 0] - Df[i0+1,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1,i2, 1 ]
			f[i0,i1,i2] += Df[i0,i1,i2, 2] - Df[i0,i1,i2+1, 2]

#		i0 is looping
#		i1 = Df.shape[1]-1
		i2 = Df.shape[2]-1
		f[i0,i1,i2] = Df[i0,i1,i2, 0] - Df[i0+1,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1,i2, 1 ]
		f[i0,i1,i2] += Df[i0,i1,i2, 2 ]

	i0 = Df.shape[0]-1
	for i1 in numba.prange(0, Df.shape[1]-1):
		for i2 in range(0, Df.shape[2]-1):
			f[i0,i1,i2] = Df[i0,i1,i2, 0]
			f[i0,i1,i2] += Df[i0,i1,i2, 1] - Df[i0,i1+1,i2, 1]
			f[i0,i1,i2] += Df[i0,i1,i2, 2] - Df[i0,i1,i2+1, 2]

#		i0 = Df.shape[0]-1
#		i1 is looping
		i2 = Df.shape[2]-1
		f[i0,i1,i2] = Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1,i2, 1] - Df[i0,i1+1,i2, 1]
		f[i0,i1,i2] += Df[i0,i1,i2, 2 ]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	for i2 in numba.prange(0, Df.shape[2]-1):
		f[i0,i1,i2] = Df[i0,i1,i2, 0]
		f[i0,i1,i2] += Df[i0,i1,i2, 1 ]
		f[i0,i1,i2] += Df[i0,i1,i2, 2] - Df[i0,i1,i2+1, 2]

#	i0 = Df.shape[0]-1
#	i1 = Df.shape[1]-1
	i2 = Df.shape[2]-1
	f[i0,i1,i2] = Df[i0,i1,i2, 0]
	f[i0,i1,i2] += Df[i0,i1,i2, 1 ]
	f[i0,i1,i2] += Df[i0,i1,i2, 2 ]


print("20/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g3_1(f,Df):
	for i0 in numba.prange(0, f.shape[0]-1):
		for i1 in range(0, f.shape[1]-1):
			for i2 in range(0, f.shape[2]-1):
				for i3 in range(f.shape[3]):
					Df[i0,i1,i2,i3, 0] = f[i0+1,i1,i2,i3] - f[i0,i1,i2,i3]
					Df[i0,i1,i2,i3, 1] = f[i0,i1+1,i2,i3] - f[i0,i1,i2,i3]
					Df[i0,i1,i2,i3, 2] = f[i0,i1,i2+1,i3] - f[i0,i1,i2,i3]

#			i0 is looping
#			i1 is looping
			i2 = f.shape[2]-1
			for i3 in range(f.shape[3]):
				Df[i0,i1,i2,i3, 0] = f[i0+1,i1,i2,i3] - f[i0,i1,i2,i3]
				Df[i0,i1,i2,i3, 1] = f[i0,i1+1,i2,i3] - f[i0,i1,i2,i3]
				Df[i0,i1,i2,i3, 2] = 0

#		i0 is looping
		i1 = f.shape[1]-1
		for i2 in range(0, f.shape[2]-1):
			for i3 in range(f.shape[3]):
				Df[i0,i1,i2,i3, 0] = f[i0+1,i1,i2,i3] - f[i0,i1,i2,i3]
				Df[i0,i1,i2,i3, 1] = 0
				Df[i0,i1,i2,i3, 2] = f[i0,i1,i2+1,i3] - f[i0,i1,i2,i3]

#		i0 is looping
#		i1 = f.shape[1]-1
		i2 = f.shape[2]-1
		for i3 in range(f.shape[3]):
			Df[i0,i1,i2,i3, 0] = f[i0+1,i1,i2,i3] - f[i0,i1,i2,i3]
			Df[i0,i1,i2,i3, 1] = 0
			Df[i0,i1,i2,i3, 2] = 0

	i0 = f.shape[0]-1
	for i1 in numba.prange(0, f.shape[1]-1):
		for i2 in range(0, f.shape[2]-1):
			for i3 in range(f.shape[3]):
				Df[i0,i1,i2,i3, 0] = 0
				Df[i0,i1,i2,i3, 1] = f[i0,i1+1,i2,i3] - f[i0,i1,i2,i3]
				Df[i0,i1,i2,i3, 2] = f[i0,i1,i2+1,i3] - f[i0,i1,i2,i3]

#		i0 = f.shape[0]-1
#		i1 is looping
		i2 = f.shape[2]-1
		for i3 in range(f.shape[3]):
			Df[i0,i1,i2,i3, 0] = 0
			Df[i0,i1,i2,i3, 1] = f[i0,i1+1,i2,i3] - f[i0,i1,i2,i3]
			Df[i0,i1,i2,i3, 2] = 0

#	i0 = f.shape[0]-1
	i1 = f.shape[1]-1
	for i2 in numba.prange(0, f.shape[2]-1):
		for i3 in range(f.shape[3]):
			Df[i0,i1,i2,i3, 0] = 0
			Df[i0,i1,i2,i3, 1] = 0
			Df[i0,i1,i2,i3, 2] = f[i0,i1,i2+1,i3] - f[i0,i1,i2,i3]

#	i0 = f.shape[0]-1
#	i1 = f.shape[1]-1
	i2 = f.shape[2]-1
	for i3 in range(f.shape[3]):
		Df[i0,i1,i2,i3, 0] = 0
		Df[i0,i1,i2,i3, 1] = 0
		Df[i0,i1,i2,i3, 2] = 0


print("21/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _gt3_1(Df,f):
	i0 = 0
	i1 = 0
	i2 = 0
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#	i0 = 0
#	i1 = 0
	for i2 in numba.prange(1, Df.shape[2]-1):
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#	i0 = 0
#	i1 = 0
	i2 = Df.shape[2]-1
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
		f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

#	i0 = 0
	for i1 in numba.prange(1, Df.shape[1]-1):
		i2 = 0
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#		i0 = 0
#		i1 is looping
		for i2 in range(1, Df.shape[2]-1):
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
				f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#		i0 = 0
#		i1 is looping
		i2 = Df.shape[2]-1
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

#	i0 = 0
	i1 = Df.shape[1]-1
	i2 = 0
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
		f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#	i0 = 0
#	i1 = Df.shape[1]-1
	for i2 in numba.prange(1, Df.shape[2]-1):
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#	i0 = 0
#	i1 = Df.shape[1]-1
	i2 = Df.shape[2]-1
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = -Df[i0,i1,i2,i3, 0]
		f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
		f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

	for i0 in numba.prange(1, Df.shape[0]-1):
		i1 = 0
		i2 = 0
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#		i0 is looping
#		i1 = 0
		for i2 in range(1, Df.shape[2]-1):
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
				f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
				f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#		i0 is looping
#		i1 = 0
		i2 = Df.shape[2]-1
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

#		i0 is looping
		for i1 in range(1, Df.shape[1]-1):
			i2 = 0
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
				f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#			i0 is looping
#			i1 is looping
			for i2 in range(1, Df.shape[2]-1):
				for i3 in range(Df.shape[3]):
					f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
					f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
					f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#			i0 is looping
#			i1 is looping
			i2 = Df.shape[2]-1
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
				f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

#		i0 is looping
		i1 = Df.shape[1]-1
		i2 = 0
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#		i0 is looping
#		i1 = Df.shape[1]-1
		for i2 in range(1, Df.shape[2]-1):
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
				f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#		i0 is looping
#		i1 = Df.shape[1]-1
		i2 = Df.shape[2]-1
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0] - Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

	i0 = Df.shape[0]-1
	i1 = 0
	i2 = 0
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#	i0 = Df.shape[0]-1
#	i1 = 0
	for i2 in numba.prange(1, Df.shape[2]-1):
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#	i0 = Df.shape[0]-1
#	i1 = 0
	i2 = Df.shape[2]-1
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 1]
		f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

#	i0 = Df.shape[0]-1
	for i1 in numba.prange(1, Df.shape[1]-1):
		i2 = 0
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#		i0 = Df.shape[0]-1
#		i1 is looping
		for i2 in range(1, Df.shape[2]-1):
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
				f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#		i0 = Df.shape[0]-1
#		i1 is looping
		i2 = Df.shape[2]-1
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1] - Df[i0,i1,i2,i3, 1]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	i2 = 0
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
		f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
		f[i0,i1,i2,i3] -= Df[i0,i1,i2,i3, 2]

#	i0 = Df.shape[0]-1
#	i1 = Df.shape[1]-1
	for i2 in numba.prange(1, Df.shape[2]-1):
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
			f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2] - Df[i0,i1,i2,i3, 2]

#	i0 = Df.shape[0]-1
#	i1 = Df.shape[1]-1
	i2 = Df.shape[2]-1
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = Df[i0-1,i1,i2,i3, 0]
		f[i0,i1,i2,i3] += Df[i0,i1-1,i2,i3, 1 ]
		f[i0,i1,i2,i3] += Df[i0,i1,i2-1,i3, 2 ]


print("22/24", end="\r")
print("23/24", end="\r")
@numba.jit(nopython=True, parallel=False, fastmath=True, cache=True)
def _g2t3_1(Df,f):
	for i0 in numba.prange(0, Df.shape[0]-1):
		for i1 in range(0, Df.shape[1]-1):
			for i2 in range(0, Df.shape[2]-1):
				for i3 in range(Df.shape[3]):
					f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0] - Df[i0+1,i1,i2,i3, 0]
					f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1] - Df[i0,i1+1,i2,i3, 1]
					f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2] - Df[i0,i1,i2+1,i3, 2]

#			i0 is looping
#			i1 is looping
			i2 = Df.shape[2]-1
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0] - Df[i0+1,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1] - Df[i0,i1+1,i2,i3, 1]
				f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2 ]

#		i0 is looping
		i1 = Df.shape[1]-1
		for i2 in range(0, Df.shape[2]-1):
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0] - Df[i0+1,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1 ]
				f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2] - Df[i0,i1,i2+1,i3, 2]

#		i0 is looping
#		i1 = Df.shape[1]-1
		i2 = Df.shape[2]-1
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0] - Df[i0+1,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1 ]
			f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2 ]

	i0 = Df.shape[0]-1
	for i1 in numba.prange(0, Df.shape[1]-1):
		for i2 in range(0, Df.shape[2]-1):
			for i3 in range(Df.shape[3]):
				f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0]
				f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1] - Df[i0,i1+1,i2,i3, 1]
				f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2] - Df[i0,i1,i2+1,i3, 2]

#		i0 = Df.shape[0]-1
#		i1 is looping
		i2 = Df.shape[2]-1
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1] - Df[i0,i1+1,i2,i3, 1]
			f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2 ]

#	i0 = Df.shape[0]-1
	i1 = Df.shape[1]-1
	for i2 in numba.prange(0, Df.shape[2]-1):
		for i3 in range(Df.shape[3]):
			f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0]
			f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1 ]
			f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2] - Df[i0,i1,i2+1,i3, 2]

#	i0 = Df.shape[0]-1
#	i1 = Df.shape[1]-1
	i2 = Df.shape[2]-1
	for i3 in range(Df.shape[3]):
		f[i0,i1,i2,i3] = Df[i0,i1,i2,i3, 0]
		f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 1 ]
		f[i0,i1,i2,i3] += Df[i0,i1,i2,i3, 2 ]


print("24/24", end="\r")
