import sys

from tools.trainer import ASTGCNTrainer
from tools.utils import load_settings

if __name__ == '__main__':
	item = sys.argv[1].lower() if len(sys.argv) >= 2 else 'debug'
	trainer = ASTGCNTrainer(load_settings(item)).run()
