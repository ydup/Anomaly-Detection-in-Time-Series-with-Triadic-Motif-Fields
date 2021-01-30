"""
Map the gen_feature.py to different nodes and cores to run them parallelly.
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python map_gen_feature.py train 10 no
"""
import sys
sys.path.append('../')
import subprocess
from lib.util import mkdir
mode= str(sys.argv[1])
nodes = int(sys.argv[2])
freq = str(sys.argv[3]) # no, mid

for idx in range(nodes):
	qsub_command = """qsub -v slice={0},nodes={1},mode={2},freq={3} -q adf gen_feature.sh""".format(*[idx, nodes, mode, freq]) # submit the job script
	exit_status = subprocess.call(qsub_command, shell=True) # upload
	if exit_status is 1:  # Check to make sure the job submitted
		print("Job {0} failed to submit".format(qsub_command))


