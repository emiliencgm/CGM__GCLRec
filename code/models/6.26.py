import os
os.system('python Adaloss.py --task gowalla --tau 0.12 --adaptive 1')
os.system('python Adaloss.py --task gowalla --tau 0.12 --adaptive 0')
os.system('python BCloss.py --task gowalla --valid 0 --temp_tau 0.12')