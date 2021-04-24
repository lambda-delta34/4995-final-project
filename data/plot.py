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
	fn = fn[3]
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
	acc_df['GPU-Laplacian+RR'] = df_new['relative_err']
	acc_df['Gaussian + Near Optimal'] = df_base['relative_err']
	#query_df['baseline query'] = df_base['query']
	acc_df.plot()
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	fn = fn1_new.split('.')
	fn = fn[1].split('/')
	fn = fn[3]
	print(fn)
	plt.savefig("./plots/acc_"+fn)
	plt.close()

# speed 

plot_speed("./speed/uniform/laplacian_tf_D_unif.csv", "./speed/uniform/near_opt_D_unif.csv", 
	 "D", "Runtime", "model comapred with D in random uniform (N=100k, L=50)")

plot_speed("./speed/normal/laplacian_tf_D.csv", "./speed/normal/near_opt_D.csv", 
	 "D", "Runtime", "model comapred with D in random normal (N=100k, L=50)")

plot_speed("./speed/normal/laplacian_tf_L.csv", "./speed/normal/near_opt_L.csv", 
	 "L", "Runtime", "model comapred with L in random normal (N=100k, D=50)")

plot_speed("./speed/uniform/laplacian_tf_L_unif.csv", "./speed/uniform/near_opt_L_unif.csv", 
	 "L", "Runtime", "model comapred with L in random uniform (N=100k, D=50)")

plot_speed("./speed/uniform/laplacian_tf_N_unif.csv", "./speed/uniform/near_opt_N_unif.csv", 
	 "N", "Runtime", "model comapred with N in random uniform (D=50, L=50)")

plot_speed("./speed/normal/laplacian_tf_N.csv", "./speed/normal/near_opt_N.csv", 
	 "N", "Runtime", "model comapred with N in random normal (D=50, L=50)")

plot_speed_2("./original_speed/laplacian_orig_D.csv", "./original_speed/laplacian_tf_orig_D.csv", 
	 "D", "Runtime", "model comapred with D under random normal (N=100k, L=50)")

plot_speed_2("./original_speed/laplacian_orig_L.csv", "./original_speed/laplacian_tf_orig_L.csv", 
	 "D", "Runtime", "model comapred with L under random normal (N=100k, D=50)")

plot_speed_2("./original_speed/laplacian_orig_N.csv", "./original_speed/laplacian_tf_orig_N.csv", 
	 "N", "Runtime", "model comapred with N under random normal (D=50, L=50)")

# accuracy 
plot_acc("./accuracy/normal/laplacian_D_acc.csv", "./accuracy/normal/near_opt_D_acc.csv", 
 	 "D", "Relative Error", "model comapred with D in random normal (N=100k, T=50)")

plot_acc("./accuracy/uniform/laplacian_D_unif_acc.csv", "./accuracy/uniform/near_opt_D_unif_acc.csv", 
 	 "D", "Relative Error", "model comapred with D in random uniform (N=100k, T=50)")

plot_acc("./accuracy/normal/laplacian_N_acc.csv", "./accuracy/normal/near_opt_N_acc.csv", 
	 "N", "Relative Error", "model comapred with N in random (D=50, T=50)")

plot_acc("./accuracy/uniform/laplacian_N_unif_acc.csv", "./accuracy/uniform/near_opt_N_unif_acc.csv", 
	 "N", "Relative Error", "model comapred with N in uniform (D=50, T=50)")
