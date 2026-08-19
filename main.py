"""Entry point for the EDX Line Scan Viewer application."""

from edx_analyzer.app import EDXApp

if __name__ == "__main__":
    app = EDXApp()
    app.mainloop()
