@echo off
echo.
echo  SAD – Статистичний Аналіз Даних
echo  Збірка виконуваного файлу...
pip install pandas numpy scipy openpyxl pyinstaller --quiet
pyinstaller --onefile --windowed --name "SAD-Статистичний-Аналіз-Даних" --icon=icon.ico main.py
echo.
echo  ГОТОВО! Файл тут: dist\SAD-Статистичний-Аналіз-Даних.exe
pause
