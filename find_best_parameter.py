import os
import os.path

rootdir="ace_en/result/"
best_recall=0.0
best_parameter=""
for parent,dirnames,filenames in os.walk(rootdir):
	for filename in filenames:
		if "result_" in filename:
			#print(filename)
			content=open(rootdir+filename).readlines()
			if len(content)>0:
				a=content[-5]
				recall=float(a[7:-1])
				if recall > best_recall:
					best_recall=recall
					best_parameter=filename
					#print(filename,recall)
print(best_parameter, best_recall)
