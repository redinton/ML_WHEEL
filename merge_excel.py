import pandas as pd 
import os

data_path = "."


def merge_data(data_path):
	files = os.listdir(data_path)
	filter_file = [x for x in files if x.endswith(".xlsx")]
	init_len = 0
	init_col = 0
	for name in filter_file:
		name = data_path+"/"+name
		try:
			data = pd.read_excel(name,header=1)
			col = list(data.columns)
			#take_col = ['Ultimate Parent','Title','Title - DWPI','Application Number','Abstract - DWPI']
			#col = take_col
			col_lengh = len(col)
			if init_len ==0:
				init_len = col_lengh
				init_col = col
				init_data = data[take_col]
			else:
				init_data = init_data.append(data[init_col],ignore_index=True)

			if not init_col == col:
				print (name)

		except Exception as e:
			print ("this file {0} has the problem".format(name))
			print (e)
			print ('\n ')
	return init_data


merge_result = merge_data(data_path)
merge_result.to_csv("merge_result.txt",index=False)
#merge_result.to_excel("merge_result.xlsx",index=False)


#read_data(data_path)
def read_data(data_path):
	files = os.listdir(data_path)
	filter_file = [x for x in files if x.endswith(".xlsx")]
	init_len = 0
	init_col = 0
	for name in filter_file:
		name = data_path+"/"+name
		try:
			data = pd.read_excel(name,header=1)
			col = list(data.columns)
			col.sort()

			col_lengh = len(col)
			if init_len ==0:
				init_len = col_lengh
				init_col = col
			if not init_col == col:
				print (name)

		except Exception as e:
			print ("this file {0} has the problem".format(name))
			print (e)
			print ('\n ')