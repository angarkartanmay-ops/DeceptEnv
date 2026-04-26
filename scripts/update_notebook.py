import json

with open('training/rl_trainer.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 0: Clone and install
nb['cells'][2]['source'] = [
    "!pip install --quiet 'transformers>=4.44' 'trl>=0.10' 'peft>=0.12' 'accelerate>=0.33' \\\n",
    "                    'datasets>=2.20' 'fastapi>=0.115' 'uvicorn[standard]>=0.30' \\\n",
    "                    'httpx>=0.27' 'matplotlib>=3.8' 'numpy>=1.26'\n",
    "\n",
    "!git clone https://huggingface.co/spaces/Jaisharma7/DeceptEnv /content/DeceptEnv\n",
    "%cd /content/DeceptEnv\n",
    "\n",
    "import sys, pathlib\n",
    "ROOT = pathlib.Path('.').resolve()\n",
    "if str(ROOT) not in sys.path:\n",
    "    sys.path.insert(0, str(ROOT))\n",
    "print('cwd:', ROOT)\n",
    "\n",
    "from google.colab import userdata\n",
    "from huggingface_hub import login\n",
    "login(token=userdata.get('HF_TOKEN'))\n"
]

with open('training/rl_trainer.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
