#!/usr/bin/env python3
"""
Beautiful Logger Utility
A colorful and structured logging utility with icons and progress indicators.
"""

from datetime import datetime
import json


class BeautifulLogger:
    """A beautiful logging utility with colors, icons, and structured output."""
    
    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bright_red': '\033[91;1m',
        'bright_green': '\033[92;1m',
        'bright_yellow': '\033[93;1m',
        'bright_blue': '\033[94;1m',
        'bright_magenta': '\033[95;1m',
        'bright_cyan': '\033[96;1m',
        'bright_white': '\033[97;1m',
    }
    
    # Icons for different message types
    ICONS = {
        'info': '📋',
        'success': '✅',
        'warning': '⚠️ ',
        'error': '❌',
        'debug': '🔍',
        'process': '⚙️ ',
        'start': '🚀',
        'complete': '🎉',
        'data': '📊',
        'tensor': '🔷',
        'operator': '⚡',
        'weight': '⚖️ ',
    }
    
    def __init__(self, enable_colors=True, enable_icons=True, enable_debug=True):
        self.enable_colors = enable_colors
        self.enable_icons = enable_icons
        self.enable_debug = enable_debug
    
    def _get_timestamp(self):
        """Get current timestamp in formatted string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _colorize(self, text, color_name):
        """Apply color to text if colors are enabled."""
        if not self.enable_colors:
            return text
        return f"{self.COLORS.get(color_name, '')}{text}{self.COLORS['reset']}"
    
    def _format_message(self, level, icon, message, color_name='white'):
        """Format a message with timestamp, level, icon, and color."""
        timestamp = self._get_timestamp()
        icon_str = self.ICONS.get(icon, '') if self.enable_icons else ''
        
        colored_level = self._colorize(f"{level:8}", color_name)
        colored_icon = self._colorize(icon_str, color_name)
        colored_message = self._colorize(message, color_name)
        
        return f"[{timestamp}] {colored_level} {colored_icon} {colored_message}"
    
    def info(self, message):
        """Print info message."""
        print(self._format_message("INFO", "info", message, "blue"))
    
    def success(self, message):
        """Print success message."""
        print(self._format_message("SUCCESS", "success", message, "green"))
    
    def warning(self, message):
        """Print warning message."""
        print(self._format_message("WARNING", "warning", message, "yellow"))
    
    def error(self, message):
        """Print error message."""
        print(self._format_message("ERROR", "error", message, "red"))
    
    def debug(self, message):
        """Print debug message if debug output is enabled."""
        if not self.enable_debug:
            return
        print(self._format_message("DEBUG", "debug", message, "cyan"))
    
    def set_debug(self, enable=True):
        """Enable or disable debug output."""
        self.enable_debug = enable
    
    def process(self, message):
        """Print process message."""
        print(self._format_message("PROCESS", "process", message, "magenta"))
    
    def start(self, message):
        """Print start message."""
        print(self._format_message("START", "start", message, "bright_green"))
    
    def complete(self, message):
        """Print complete message."""
        print(self._format_message("COMPLETE", "complete", message, "bright_green"))
    
    def data(self, message):
        """Print data message."""
        print(self._format_message("DATA", "data", message, "bright_cyan"))
    
    def tensor(self, message):
        """Print tensor-related message."""
        print(self._format_message("TENSOR", "tensor", message, "bright_blue"))
    
    def operator(self, message):
        """Print operator-related message."""
        print(self._format_message("OPERATOR", "operator", message, "bright_magenta"))
    
    def weight(self, message):
        """Print weight-related message."""
        print(self._format_message("WEIGHT", "weight", message, "bright_yellow"))
    
    def separator(self, char='=', length=60):
        """Print a separator line."""
        print(self._colorize(char * length, "white"))
    
    def box(self, title, content, title_color="bright_cyan"):
        """Print content in a nice box."""
        self.separator()
        title_line = self._colorize(f" {title} ", title_color)
        padding = (60 - len(title) - 2) // 2
        print(self._colorize(' ' * padding + title_line + ' ' * padding, title_color))
        self.separator()
        
        if isinstance(content, (dict, list)):
            # Pretty print JSON data
            formatted_content = json.dumps(content, indent=2, ensure_ascii=False)
            print(self._colorize(formatted_content, "white"))
        else:
            print(self._colorize(str(content), "white"))
        
        self.separator()
    
    def progress(self, current, total, prefix="Progress", width=40):
        """Display a progress bar."""
        percentage = (current / total) * 100
        filled_length = int(width * current // total)
        bar = '█' * filled_length + '-' * (width - filled_length)
        
        progress_line = f"{prefix}: |{bar}| {percentage:.1f}% ({current}/{total})"
        print(self._colorize(progress_line, "bright_green"), end='\r')
        
        if current == total:
            print()  # New line when complete
