from fastapi import FastAPI
import importlib
import pkgutil
from app.routers import __path__ as routers_path

app = FastAPI()

# Автоматически импортируем все модули из `routers/`
for _, module_name, _ in pkgutil.iter_modules(routers_path):
    module = importlib.import_module(f"app.routers.{module_name}")
    app.include_router(module.router)
