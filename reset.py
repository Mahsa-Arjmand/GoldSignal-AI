import shutil
from pathlib import Path

models_dir = Path('models')
if models_dir.exists():
    shutil.rmtree(models_dir)
    print(" پوشه models کاملاً حذف شد")

models_dir.mkdir(exist_ok=True)
print(" پوشه models خالی ایجاد شد")

import os
cache_dir = Path(os.environ.get('USERPROFILE', '')) / '.streamlit' / 'cache'
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(" Streamlit cache حذف شد")