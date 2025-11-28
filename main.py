import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QTableWidget,
    QTableWidgetItem, QMessageBox, QTextEdit, QDialog, QVBoxLayout
)
from PyQt5.QtCore import Qt
from analysis import check_normality  # твоя функція перевірки нормальності


class ResultDialog(QDialog):
    """Вікно для відображення результатів аналізу"""
    def __init__(self, title, text):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(800, 600)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setText(text)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAD - Статистичний аналіз даних")
        self.resize(1000, 600)

        # Початкова таблиця 10x7
        self.table = QTableWidget(10, 7)
        headers = ["Фактор А", "Фактор В", "Фактор С", "Повт.1", "Повт.2", "Повт.3", "Повт.4"]
        self.table.setHorizontalHeaderLabels(headers)
        self.setCentralWidget(self.table)

        # Робимо всі клітинки редагованими
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

        # Аналіз → автоматичний запуск при натисканні
        analysis_menu.triggered.connect(self.run_analysis)

        # Про програму
        about_action = QAction("Інформація", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Додаємо підтримку Ctrl+C / Ctrl+V
        self.table.installEventFilter(self)

    # Дії з таблицею
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
                item = QTableWidgetItem(val)
                item.setFlags(item.flags() | Qt.ItemIsEditable)
                self.table.setItem(r + i, c + j, item)

    # Автоматичний запуск аналізу
    def run_analysis(self):
        # Збираємо дані
        data = []
        for i in range(self.table.rowCount()):
            row_values = []
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                if item and item.text().strip():
                    try:
                        row_values.append(float(item.text()))
                    except ValueError:
                        row_values.append(None)
                else:
                    row_values.append(None)
            data.append(row_values)

        if not any(any(v is not None for v in row) for row in data):
            QMessageBox.warning(self, "Помилка", "Дані відсутні або некоректні")
            return

        # Виконуємо перевірку нормальності (Shapiro-Wilk) для всіх чисел
        flat_data = [v for row in data for v in row if v is not None]
        result = check_normality(flat_data)

        # Формуємо текст результату
        report = f"Перевірка нормальності залишків (Shapiro-Wilk): W = {result['W']:.4f}, p = {result['p']:.4f}\n\n"
        report += "Дані:\n"
        for row in data:
            report += "\t".join(str(v) if v is not None else "" for v in row) + "\n"

        # Показуємо результат у вікні
        dlg = ResultDialog("Результат аналізу", report)
        dlg.exec_()

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
