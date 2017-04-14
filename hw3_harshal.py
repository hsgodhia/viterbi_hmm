import numpy as np, re

MAX_SEGMENT_LENGTH = 5

def feature_function(y_cur, y_prev, t, u, x):
	active_features = {}
	d = u - t + 1
	
	if y_cur == 'NAME':
		status = True

		for i in range(t, u + 1):
			matchobj = re.match(r'[A-Z][a-z]*', x[i])
			if matchobj is None:
				status = False
				break
			elif len(matchobj.group()) <= 0:
				status = False
				break
		
		if status:
			active_features['all_name'] = 1
			#further check the length of the NAME sequence
			if d == 2:
				active_features['name_len_2'] = 1

			elif d == 3:
				active_features['name_len_3'] = 1
			
			elif d == 4:
				active_features['name_len_4'] = 1
			
			elif d == 5:
				active_features['name_len_5'] = 1
			
	#the below form the segment transition features 
	if y_cur == 'NAME' and y_prev == 'NAME' and t > 1 and u < len(x) - 1:
		active_features['name_to_name'] = 1

	if y_cur == 'O' and y_prev == 'O' and t > 1 and u < len(x) - 1:
		active_features['o_to_o'] = 1

	if y_cur == 'O' and y_prev == 'NAME' and t > 1 and u < len(x) - 1:
		active_features['name_to_o'] = 1

	if y_cur == 'NAME' and y_prev == 'O' and t > 1 and u < len(x) - 1:
		active_features['o_to_name'] = 1

	if (y_cur == 'O' or y_cur == 'NAME' ) and y_prev == 'START' and t == 1:
		active_features['start_to_o_or_name'] = 1

	if (y_prev == 'O' or y_prev == 'NAME' ) and y_cur == 'END' and u == len(x) - 1:
		active_features['o_or_name_to_end'] = 1

	#the below is a feature for O entity, activating for unit length 0 segments
	if y_cur == 'O' and d == 1:
		matchobj = re.match(r'[a-z]*', x[t])
		if matchobj and len(matchobj.group()) > 1:
			active_features['o_len_1'] = 1

	return active_features

def viterbi(toks, tags, weights):
	#V has dimension 
	V = np.zeros((len(toks), len(tags)), dtype=np.dtype(int))
	prev = np.zeros((len(toks), len(tags)), dtype=np.dtype(int))
	vd = np.zeros((len(toks), len(tags)), dtype=np.dtype(int))
	#assume sequence of words start at position 1
	L = MAX_SEGMENT_LENGTH
	#set the maximum segment length as 3
	for y in range(0, len(tags)):
		V[0][y] = 0

	for i in range(1, len(toks) - 1):
		for y in range(0, len(tags)):
			maxv, max_y_p, max_d = 0, 0, 0
			for d in range(1, L + 1):
				for y_p in range(0, len(tags)):
					active_features = feature_function(tags[y], tags[y_p], i - d + 1, i, toks)

					#since numpy allows negative index, we catch that and assign negative infinity
					if i - d < 0:
						dp_val = -100000000
					else:
						dp_val = V[i - d][y_p]

					curv =  dp_val + dot_prod(weights, active_features)
					if curv > maxv:
						maxv = curv
						max_y_p = y_p
						max_d = d

			V[i][y] = maxv
			prev[i][y] = max_y_p
			vd[i][y] = max_d

	maxy, maxv = 0, 0
	endpos = len(toks) - 2
	maxv = np.argmax(V, axis = 1)
	maxv_y = maxv[endpos]
	print("Viterbi Score: {0}".format(V[endpos, maxv_y]))
	print("Segmentation is: ", end='')
	seg = []
	vtag = []
	while endpos > 0:
		vtag.append(tags[maxv_y])
		t = vd[endpos][maxv_y]
		startpos = endpos - t + 1
		seg.append((startpos, endpos))
		maxv_y = prev[endpos][maxv_y]
		endpos = startpos - 1
		
	while len(seg) > 0:
		print("{0} {1}".format(vtag.pop(), seg.pop()), end=' ')

def dot_prod(weights, feature_function):
	res = 0
	for k in feature_function:
		res += weights[k]*feature_function[k]
	return res

def main():
	#START and END are special tokens that are only used for position i=1 and i =n respectively
	#start token @ is added to the sentence, end token $ is added to the end of the sentence
	#tokens of the sentence, assume delimiter is space for now

	tags = ['O', 'NAME', 'START', 'END']
	weights = {"o_or_name_to_end":2, "start_to_o_or_name":2, 'name_len_2':29, 'name_len_3':43, 'name_len_4':73, 'name_len_5':79, 'o_to_o':11, 'o_len_1':13, 'all_name':13, 'name_to_name':-3, 'o_to_name':5, 'name_to_o':5}
	
	tests = \
	["Zero Dark Thirty is a good movie", \
	"i went to Blue Heron", \
	"Santa clara Convention Center and Mr. Harshal to Bora Bora", \
	"she lives in New York City", \
	"we went to Santa Clara Convention Center and Mr. Harshal to Bora Bora", \
	"Carnegie Mellon University is good for Computer Science"]

	for test in tests:
		toks = []
		toks.append("@")
		toks.extend(test.split(" "))
		toks.append("$")
		print("Test:" + test)
		viterbi(toks, tags, weights)
		print("\n")
		input("Click 'any key' to go to next test case....\n")

main()