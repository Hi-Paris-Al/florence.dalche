import os

print('Running eps_KRR...')
os.system('python Yeast_results.py e_krr')

print('Running eps_SVR...')
os.system('python Yeast_results.py e_svr')

print('Running k_Huber...')
os.system('python Yeast_results.py k_huber')
