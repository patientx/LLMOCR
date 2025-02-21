import sys
import os
import time  # Import the time module
from typing import Optional, List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, 
    QLineEdit, QLabel, QFileDialog, QProgressBar, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QClipboard
from koboldapi import KoboldAPICore, ImageProcessor
import pyperclip  # Fallback for clipboard functionality

class LLMProcessor:
    def __init__(self, api_url, system_instruction, instruction, max_length=256):
        self.instruction = instruction
        self.system_instruction = system_instruction
        self.max_length = max_length  # Store max_length
        self.core = KoboldAPICore(api_url=api_url)
        self.image_processor = ImageProcessor(max_dimension=384)
        print("-----------------------------------------------------------------")
        print("*** Processing... ")
        
    def process_file(self, file_path: str) -> tuple[Optional[str], str]:
        """Process an image file and return (result, output_path)"""
        encoded_image, output_path = self.image_processor.process_image(str(file_path))
        result = self.core.wrap_and_generate(
            instruction=self.instruction, 
            system_instruction=self.system_instruction, 
            images=[encoded_image],
            max_length=self.max_length,  # Use dynamic max_length
            top_p=0.95,
            top_k=0,
            temp=0.6,
            rep_pen=1.1,
            min_p=0.1
        )
        print("*** Output: ")
        print(f"\n{result}\n")
        return result, output_path
        
    def save_result(self, result: str, output_path: str) -> bool:
        """Save the processing result to a file"""
        txt_output_path = os.path.splitext(output_path)[0] + ".txt"
        try:
            with open(txt_output_path, "w", encoding="utf-8") as output_file:
                output_file.write(result)
            return True
        except Exception as e:
            print(f"Error saving to {txt_output_path}: {e}")
            return False

class ProcessingThread(QThread):
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result_ready = pyqtSignal(str)  # New signal for clipboard operations
    
    def __init__(self, processor: LLMProcessor, files: List[str]):
        super().__init__()
        self.processor = processor
        self.files = files
        self.start_time = None  # Store the start time
        
    def run(self):
        try:
            self.start_time = time.time()  # Record the start time
            for i, file_path in enumerate(self.files):
                result, output_path = self.processor.process_file(file_path)
                if result:
                    self.processor.save_result(result, output_path)
                    self.result_ready.emit(result)  # Emit result for clipboard
                self.progress.emit(i + 1, len(self.files))
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LLM Image Processor")
        self.setMinimumWidth(600)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # API URL
        layout.addWidget(QLabel("API URL:"))
        self.api_url = QLineEdit("http://localhost:5001")
        layout.addWidget(self.api_url)
        
        layout.addWidget(QLabel("System Instruction:"))
        self.system_instruction = QLineEdit("You are a helpful image captioner.")
        layout.addWidget(self.system_instruction)
        
        # Instruction selection
        layout.addWidget(QLabel("Instruction:"))
        self.instruction_combo = QComboBox()
        self.instruction_combo.addItems([
            "Describe the image in detail. Be specific.",
            "Provide a brief description of the image.",
            "Write a descriptive caption for this image in a formal tone within 250 words.",
            "Write a long descriptive caption for this image in a formal tone.",
            "Write a descriptive caption for this image in a formal tone.",
            "Write a stable diffusion prompt for this image.",
            "Write a stable diffusion prompt for this image within 250 words.",
            "Write a MidJourney prompt for this image.",
            "Write a list of Booru tags for this image.",
            "Write a list of Booru-like tags for this image.",
            "Analyze this image like an art critic would.",
            "Describe the main objects and their relationships in the image."
        ])
        layout.addWidget(self.instruction_combo)
        
        # Max Length Input
        layout.addWidget(QLabel("Max Length:"))
        self.max_length_input = QLineEdit("256")  # Default value
        layout.addWidget(self.max_length_input)
        
        # File selection
        self.file_button = QPushButton("Select Images")
        self.file_button.clicked.connect(self.select_files)
        layout.addWidget(self.file_button)
        
        # Selected files display
        self.files_label = QLabel("No files selected")
        layout.addWidget(self.files_label)
        
        # Progress bar
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        
        # Process button
        self.process_button = QPushButton("Process Images")
        self.process_button.clicked.connect(self.process_files)
        self.process_button.setEnabled(False)
        layout.addWidget(self.process_button)
        
        self.selected_files = []
        
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "*")
        if files:
            self.selected_files = files
            self.files_label.setText(f"Selected {len(files)} images")
            self.process_button.setEnabled(True)
    
    def process_files(self):
        if not self.selected_files:
            return
            
        self.process_button.setEnabled(False)
        self.file_button.setEnabled(False)
        
        processor = LLMProcessor(
            self.api_url.text(),
            self.system_instruction.text(),
            self.instruction_combo.currentText(),
            max_length=int(self.max_length_input.text()) if self.max_length_input.text().isdigit() else 256
        )
        
        self.thread = ProcessingThread(processor, self.selected_files)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.processing_finished)
        self.thread.error.connect(self.processing_error)
        self.thread.result_ready.connect(self.copy_to_clipboard)  # Connect result_ready signal
        self.thread.start()
    
    def update_progress(self, current, total):
        self.progress.setValue(int((current / total) * 100))
    
    def processing_finished(self):
        self.process_button.setEnabled(True)
        self.file_button.setEnabled(True)
        self.files_label.setText(f"Selected {len(self.selected_files)} images")
        self.progress.setValue(0)
        
        # Calculate and print the elapsed time
        if self.thread.start_time:
            elapsed_time = time.time() - self.thread.start_time
            print("*** Output copied to clipboard and saved as txt file  ***")
            print(f"*** Processing completed in {elapsed_time:.2f} seconds ***")
            print(" ")
    
    def processing_error(self, error_msg):
        self.process_button.setEnabled(True)
        self.file_button.setEnabled(True)
        self.files_label.setText(f"Error: {error_msg}")
        self.progress.setValue(0)

    def copy_to_clipboard(self, text: str):
        """Copy text to the clipboard using pyperclip"""
        try:
            pyperclip.copy(text)
        except Exception as e:
            print(f"Failed to copy to clipboard: {e}")

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
