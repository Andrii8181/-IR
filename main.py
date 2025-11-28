import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QTableWidget,
    QTableWidgetItem, QMessageBox, QInputDialog, QTextEdit, QDialog, QVBoxLayout
)
from PyQt5.QtCore import Qt
import numpy as np
from scipy.stats import shapiro

class ResultWindow(QDialog):
    def __init__(self, text):
        super().__init__()
        self.setWindowTitle("Результати аналізу")
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(text)
        self.text_edit.setReadOnly(False)  # Можна копіювати
        layout.addWidget(self.text_edit)
        self.setLayout(layout)
        self.resize(1000, 600)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAD - Статистичний аналіз даних")
        self.resize(1200, 600)

        # Таблиця 10x7
        self.table = QTableWidget(10, 7)
        self.col_names = ["Фактор А", "Фактор В", "Фактор С", "Повт.1", "Повт.2", "Повт.3", "Повт.4"]
        self.table.setHorizontalHeaderLabels(self.col_names)
        self.setCentralWidget(self.table)

        # Редагування клітинок
        for i in range(self.table.rowCount()):
            for j in range(self.table.columnCount()):
                item = QTableWidgetItem("")
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

        # Меню
        menubar = self.menuBar()
        table_menu = menubar.addMenu("Таблиця")
        analysis_menu = menubar.addMenu("Аналіз даних")
        help_menu = menubar.addMenu("Про програму")

        # Таблиця → додати / видалити
        add_row = QAction("Додати рядок", self)
        add_row.triggered.connect(self.add_row)
        table_menu.addAction(add_row)

        del_row = QAction("Видалити рядок", self)
        del_row.triggered.connect(self.delete_row)
        table_menu.addAction(del_row)

        add_col = QAction("Додати стовпчик", self)
        add_col.triggered.connect(self.add_col)
        table_menu.addAction(add_col)

        del_col = QAction("Видалити стовпчик", self)
        del_col.triggered.connect(self.delete_col)
        table_menu.addAction(del_col)

        # Аналіз → Запуск аналізу
        analyze_action = QAction("Аналіз", self)
        analyze_action.triggered.connect(self.run_analysis)
        analysis_menu.addAction(analyze_action)

        # Про програму
        about_action = QAction("Інформація", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Підтримка Ctrl+C / Ctrl+V
        self.table.installEventFilter(self)

    # Додавання / видалення рядків та колонок
    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        for j in range(self.table.columnCount()):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.table.setItem(row, j, item)

    def delete_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def add_col(self):
        col = self.table.columnCount()
        self.table.insertColumn(col)
        self.table.setHorizontalHeaderItem(col, QTableWidgetItem(f"Фактор {col+1}"))
        for i in range(self.table.rowCount()):
            item = QTableWidgetItem("")
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.table.setItem(i, col, item)

    def delete_col(self):
        col = self.table.currentColumn()
        if col >= 0:
            self.table.removeColumn(col)

    # Ctrl+C / Ctrl+V
    def eventFilter(self, source, event):
        if source == self.table:
            if event.type() == event.KeyPress:
                if event.matches(event.Copy):
                    self.copy_selection()
                    return True
                elif event.matches(event.Paste):
                    self.paste_selection()
                    return True
        return super().eventFilter(source, event)

    def copy_selection(self):
        selected = self.table.selectedRanges()
        if selected:
            r = selected[0]
            text = ""
            for i in range(r.rowCount()):
                row_data = []
                for j in range(r.columnCount()):
                    item = self.table.item(r.topRow() + i, r.leftColumn() + j)
                    row_data.append(item.text() if item else "")
                text += "\t".join(row_data) + "\n"
            QApplication.clipboard().setText(text)

    def paste_selection(self):
        text = QApplication.clipboard().text()
        rows = text.splitlines()
        r = self.table.currentRow()
        c = self.table.currentColumn()
        for i, row in enumerate(rows):
            for j, val in enumerate(row.split("\t")):
                if r + i < self.table.rowCount() and c + j < self.table.columnCount():
                    item = QTableWidgetItem(val)
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                    self.table.setItem(r + i, c + j, item)

    # Аналіз
    def run_analysis(self):
        # Збираємо дані
        data = []
        for i in range(self.table.rowCount()):
            row = []
            for j in range(3, self.table.columnCount()):
                item = self.table.item(i, j)
                if item and item.text().strip():
                    try:
                        row.append(float(item.text()))
                    except ValueError:
                        row.append(np.nan)
                else:
                    row.append(np.nan)
            data.append(row)
        numeric_data = np.array(data).flatten()
        numeric_data = numeric_data[~np.isnan(numeric_data)]
        if len(numeric_data) == 0:
            QMessageBox.warning(self, "Помилка", "Дані відсутні або некоректні")
            return

        W, p = shapiro(numeric_data)

        # Створюємо результат як текст
        result_text = f"Р Е З У Л Ь Т А Т И   Т Р И Ф А К Т О Р Н О Г О   Д И С П Е Р С І Й Н О Г О   А Н А Л І З У\n\n"
        result_text += f"Перевірка нормальності залишків (Shapiro-Wilk): {'нормальний' if p>0.05 else 'ненормальний'} (W={W:.4f}, p={p:.4f})\n\n"
        result_text += "Джерело варіації, сума квадратів, ступені свободи, середній квадрат, F...\n"
        result_text += "(Цей блок можна адаптувати для реальних одно-, дво-, трифакторних обчислень)\n"

        # Відкриваємо вікно з результатом
        res_win = ResultWindow(result_text)
        res_win.exec_()

    # Про програму
    def show_about(self):
        QMessageBox.information(self, "Про програму",
                                "SAD - Статистичний аналіз даних\n"
                                "Розробник: ....\n"
                                "Кафедра плодівництва і виноградарства УНУ")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
