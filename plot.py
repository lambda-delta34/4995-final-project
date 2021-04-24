import numpy 
import pandas as pd
import matplotlib.pyplot as plt
 
def plot_speed(fn1_new, fn2_baseline, x_label, y_label, title):
	df_new = pd.read_csv(fn1_new)
	df_base = pd.read_csv(fn2_baseline)
 
	pre_df = pd.DataFrame()
	pre_df['Laplacian+RR LSH'] = df_new['pre']
	pre_df['Gaussian+NearOpt'] = df_base['pre']
	#query_df['baseline query'] = df_base['query']
	pre_df.plot()
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label+ " (s)")
 
	#plt.show()
	fn = fn1_new.split('.')
	fn = fn[1].split('/')
	fn = fn[2]
	plt.savefig("./plots/pre_"+fn)
	plt.close()

	query_df = pd.DataFrame()
	query_df['Laplacian+RR LSH'] = df_new['query']
	query_df['Gaussian+NearOpt'] = df_base['query']
	#query_df['baseline query'] = df_base['query']
	query_df.plot()
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label+ " (s)")
	#plt.show()
	plt.savefig("./plots/query_"+fn)
	plt.close()


def plot_speed_2(fn1_new, fn2_baseline, x_label, y_label, title):
	df_new = pd.read_csv(fn1_new)
	df_base = pd.read_csv(fn2_baseline)
 
	pre_df = pd.DataFrame()
	pre_df['original Laplacian+RR pre'] = df_new['pre']
	pre_df['GPU-Laplacian+RR pre'] = df_base['pre']
	#query_df['baseline query'] = df_base['query']
	pre_df.plot()
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label+ " (s)")
 
	#plt.show()
	fn = fn1_new.split('.')
	fn = fn[1].split('/')
	fn = fn[2]
	plt.savefig("./plots2/pre_"+fn)
	plt.close()

	query_df = pd.DataFrame()
	query_df['original Laplacian+RR query'] = df_new['query']
	query_df['GPU-Laplacian+RR query'] = df_base['query']
	#query_df['baseline query'] = df_base['query']
	query_df.plot()
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label+ " (s)")
	#plt.show()
	plt.savefig("./plots2/query_"+fn)
	plt.close()


def plot_acc(fn1_new, fn2_baseline, x_label, y_label, title):
	df_new = pd.read_csv(fn1_new)
	df_base = pd.read_csv(fn2_baseline) 

	acc_df = pd.DataFrame()
	acc_df['new model relative err'] = df_new['relative_err']
	acc_df['baseline relative err'] = df_base['relative_err']
	#query_df['baseline query'] = df_base['query']
	acc_df.plot()
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	fn = fn1_new.split('.')
	fn = fn[1].split('/')
	fn = fn[2]
	print(fn)
	plt.savefig("./plots/acc_"+fn)
	plt.close()

# speed 

plot_speed("./speed/uniform/laplacian_tf_D_unif.csv", "./speed/near_opt_D_unif.csv", 
	 "D", "Runtime", "model comapred with D in uniform")

plot_speed("./speed/normal/laplacian_tf_D.csv", "./speed/near_opt_D.csv", 
	 "D", "Runtime", "model comapred with D in random")

plot_speed("./speed/normal/laplacian_tf_L.csv", "./speed/near_opt_L.csv", 
	 "L", "Runtime", "model comapred with K in random")

plot_speed("./speed/uniform/laplacian_tf_L_unif.csv", "./speed/near_opt_L_unif.csv", 
	 "L", "Runtime", "model comapred with L in uniform")

plot_speed("./speed/uniform/laplacian_tf_N_unif.csv", "./speed/near_opt_N_unif.csv", 
	 "N", "Runtime", "model comapred with N in uniform")

plot_speed("./speed/normal/laplacian_tf_N.csv", "./speed/near_opt_N.csv", 
	 "N", "Runtime", "model comapred with N in random")

plot_speed_2("./original_speed/normal/laplacian_orig_D.csv", "./original_speed/laplacian_tf_orig_D.csv", 
	 "D", "Runtime", "model comapred with D")

plot_speed_2("./original_speed/normal/laplacian_orig_L.csv", "./original_speed/laplacian_tf_orig_L.csv", 
	 "D", "Runtime", "model comapred with L")

plot_speed_2("./original_speed/normal/laplacian_orig_N.csv", "./original_speed/laplacian_tf_orig_N.csv", 
	 "N", "Runtime", "model comapred with N ")

# accuracy 
# plot_acc("./accuracy/laplacian_D_acc.csv", "./accuracy/near_opt_D_acc.csv", 
# 	 "D", "Relative Error", "model comapred with D in random")

# plot_acc("./accuracy/laplacian_D_unif_acc.csv", "./accuracy/near_opt_D_unif_acc.csv", 
# 	 "D", "Relative Error", "model comapred with D in uniform")

# plot_acc("./accuracy/laplacian_N_acc.csv", "./accuracy/near_opt_N_acc.csv", 
# 	 "N", "Relative Error", "model comapred with N in random")

# plot_acc("./accuracy/laplacian_N_unif_acc.csv", "./accuracy/near_opt_N_unif_acc.csv", 
# 	 "N", "Relative Error", "model comapred with N in uniform")
