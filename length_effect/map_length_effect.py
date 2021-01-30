"""
Map the parallel_flex.py to different nodes and cores to run them parallelly.
Author: Yadong Zhang
E-mail: zhangyadong@stu.xjtu.edu.cn

Demo:
$ python3 map_length_effect.py 10
"""
import sys
sys.path.append('../')
import subprocess
from lib.util import mkdir

nodes = int(sys.argv[1])
for idx in range(nodes):
	mkdir('slice/{0}/'.format(idx))
	qsub_command = """qsub -v slice={0},nodes={1} -q adf length_effect.sh""".format(*[idx, nodes])
	exit_status = subprocess.call(qsub_command, shell=True) # upload
	if exit_status is 1:  # Check to make sure the job submitted
		print("Job {0} failed to submit".format(qsub_command))


